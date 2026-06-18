#!/usr/bin/env python3
"""
Kubernetes Control Plane Metrics Exporter
Collects metrics from local Kubernetes control plane component endpoints
(/metrics on apiserver, etcd, scheduler, controller-manager) and via kubectl,
then exposes them in Prometheus format on a configurable HTTP port.

Intended to run directly on a Kubernetes control plane node as root.

Auth strategy (in priority order):
  1. For apiserver, scheduler, controller-manager:
     Use `kubectl get --raw /metrics` which reads auth from kubeconfig automatically.
  2. For etcd:
     Use auto-detected PKI certs from /etc/kubernetes/pki/etcd/ OR
     explicit --etcd-ca-cert / --etcd-client-cert / --etcd-client-key flags.
  3. Fallback: raw HTTPS with --ca-cert / --client-cert / --client-key flags.

Usage:
    python3 k8s_control_plane_exporter.py [options]

Options:
    --port              HTTP port (default: 9419)
    --interval          Collection interval in seconds (default: 30)
    --kubeconfig        Path to kubeconfig (auto-detected if not set)
    --apiserver-url     API server URL (default: https://127.0.0.1:6443)
    --etcd-ca-cert      etcd CA cert (default: auto-detect from PKI dir)
    --etcd-client-cert  etcd client cert (default: auto-detect)
    --etcd-client-key   etcd client key (default: auto-detect)
    --ca-cert           Generic CA cert for all HTTPS calls
    --client-cert       Generic client cert for all HTTPS calls
    --client-key        Generic client key for all HTTPS calls
    --token-file        Bearer token file (fallback)
"""

import argparse
import http.server
import json
import logging
import os
import re
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("k8s_cp_exporter")

metrics_output = "# K8s control plane metrics not yet collected\n"
metrics_lock = threading.Lock()

METRIC_PREFIX = "k8s"
DEFAULT_PORT = 9419
DEFAULT_INTERVAL = 30

# Standard K8s PKI paths on kubeadm-provisioned clusters
K8S_PKI_DIR = "/etc/kubernetes/pki"
ETCD_PKI_DIR = "/etc/kubernetes/pki/etcd"

# Candidate etcd client cert/key pairs (tried in order)
ETCD_CERT_CANDIDATES = [
    # healthcheck-client is the least-privileged cert for /metrics reads
    (f"{ETCD_PKI_DIR}/healthcheck-client.crt", f"{ETCD_PKI_DIR}/healthcheck-client.key"),
    # peer cert works too on most clusters
    (f"{ETCD_PKI_DIR}/peer.crt", f"{ETCD_PKI_DIR}/peer.key"),
    # server cert as last resort
    (f"{ETCD_PKI_DIR}/server.crt", f"{ETCD_PKI_DIR}/server.key"),
]
ETCD_CA_CANDIDATES = [
    f"{ETCD_PKI_DIR}/ca.crt",
    f"{K8S_PKI_DIR}/ca.crt",
]


def _find_etcd_certs() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect etcd TLS credentials from common kubeadm PKI paths.
    Returns (ca_cert, client_cert, client_key) — any may be None if not found.
    """
    ca = next((p for p in ETCD_CA_CANDIDATES if os.path.exists(p)), None)
    for cert, key in ETCD_CERT_CANDIDATES:
        if os.path.exists(cert) and os.path.exists(key):
            logger.info(f"Auto-detected etcd certs: cert={cert}")
            return ca, cert, key
    return ca, None, None


class K8sMetricsCollector:
    def __init__(
        self,
        kubeconfig: Optional[str] = None,
        apiserver_url: str = "https://127.0.0.1:6443",
        # Generic TLS (fallback for raw HTTPS calls)
        ca_cert: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        token_file: Optional[str] = None,
        # etcd-specific TLS
        etcd_ca_cert: Optional[str] = None,
        etcd_client_cert: Optional[str] = None,
        etcd_client_key: Optional[str] = None,
    ):
        self.kubeconfig = kubeconfig
        self.apiserver_url = apiserver_url.rstrip("/")
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key
        self.token: Optional[str] = None

        # Load bearer token if provided (used only for apiserver raw fallback)
        if token_file and os.path.exists(token_file):
            with open(token_file) as f:
                self.token = f.read().strip()
        if not self.token:
            sa_token = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(sa_token):
                with open(sa_token) as f:
                    self.token = f.read().strip()

        # etcd TLS: prefer explicit args, then auto-detect
        if etcd_ca_cert or etcd_client_cert or etcd_client_key:
            self.etcd_ca = etcd_ca_cert
            self.etcd_cert = etcd_client_cert
            self.etcd_key = etcd_client_key
        else:
            self.etcd_ca, self.etcd_cert, self.etcd_key = _find_etcd_certs()

        if self.etcd_cert:
            logger.info(f"etcd client cert: {self.etcd_cert}")
        else:
            logger.warning(
                "No etcd client cert found — etcd metrics will be unavailable. "
                "Pass --etcd-client-cert and --etcd-client-key to fix this."
            )

    # ------------------------------------------------------------------
    # kubectl helper: primary method for apiserver/scheduler/CM metrics
    # ------------------------------------------------------------------

    def _kubectl_raw(self, path: str, timeout: int = 20) -> Optional[str]:
        """
        Run `kubectl get --raw <path>` which uses the kubeconfig for auth.
        This is the correct way to access /metrics on apiserver, scheduler,
        and controller-manager — kubectl handles TLS + token/cert automatically.
        """
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["get", "--raw", path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            logger.debug(f"kubectl get --raw {path} failed (rc={result.returncode}): {result.stderr[:300]}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"kubectl get --raw {path} timed out")
            return None
        except Exception as e:
            logger.debug(f"kubectl get --raw {path} error: {e}")
            return None

    def _kubectl(self, args: List[str], timeout: int = 30) -> Optional[Dict]:
        """Run kubectl with JSON output and return parsed result."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += args + ["-o", "json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            logger.debug(f"kubectl {args[0]} failed: {result.stderr[:200]}")
            return None
        except Exception as e:
            logger.debug(f"kubectl {args[0]} error: {e}")
            return None

    # ------------------------------------------------------------------
    # Raw HTTPS helper: used for etcd (which kubectl can't proxy)
    # ------------------------------------------------------------------

    def _make_ssl_context(
        self,
        ca_cert: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
    ) -> ssl.SSLContext:
        """Build an SSL context. Always disables hostname check for localhost."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        if ca_cert and os.path.exists(ca_cert):
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_verify_locations(ca_cert)
        else:
            ctx.verify_mode = ssl.CERT_NONE
        if client_cert and client_key:
            if os.path.exists(client_cert) and os.path.exists(client_key):
                ctx.load_cert_chain(client_cert, client_key)
        return ctx

    def _https_get(
        self,
        url: str,
        ca_cert: Optional[str] = None,
        client_cert: Optional[str] = None,
        client_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        timeout: int = 15,
    ) -> Optional[str]:
        """Raw HTTPS GET with explicit cert/token control (no magic)."""
        try:
            req = urllib.request.Request(url)
            if bearer_token:
                req.add_header("Authorization", f"Bearer {bearer_token}")
            ctx = self._make_ssl_context(ca_cert, client_cert, client_key)
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            logger.debug(f"HTTP {e.code} from {url}: {e.reason}")
            return None
        except Exception as e:
            logger.debug(f"HTTPS GET {url} failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Prometheus text parser
    # ------------------------------------------------------------------

    def _parse_prometheus_metrics(self, text: str) -> Dict[str, List[Tuple]]:
        result: Dict[str, List[Tuple]] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+([\d.eE+\-]+(?:NaN|Inf)?)', line)
            if not m:
                continue
            name, labels_str, value_str = m.group(1), m.group(2), m.group(3)
            try:
                value = float(value_str)
            except ValueError:
                continue
            labels: Dict[str, str] = {}
            if labels_str:
                for lm in re.finditer(r'(\w+)="([^"]*)"', labels_str):
                    labels[lm.group(1)] = lm.group(2)
            result.setdefault(name, []).append((labels, value))
        return result

    def _histogram_quantile(
        self,
        parsed: Dict,
        metric_base: str,
        quantile: float,
    ) -> Dict[str, float]:
        """Approximate quantile from histogram bucket data."""
        results: Dict[str, float] = {}
        bucket_key = metric_base + "_bucket"
        if bucket_key not in parsed:
            return results

        groups: Dict[str, List[Tuple[float, float]]] = {}
        for labels, value in parsed.get(bucket_key, []):
            le = labels.get("le", "+Inf")
            group_label = labels.get("verb", labels.get("queue", "total"))
            groups.setdefault(group_label, [])
            try:
                le_val = float(le) if le != "+Inf" else float("inf")
            except ValueError:
                continue
            groups[group_label].append((le_val, value))

        for key, buckets in groups.items():
            buckets.sort(key=lambda x: x[0])
            if not buckets:
                continue
            total = buckets[-1][1]
            if total == 0:
                continue
            target = quantile * total
            prev_count, prev_le = 0.0, 0.0
            for le_val, count in buckets:
                if count >= target:
                    if count == prev_count:
                        results[key] = le_val
                    else:
                        frac = (target - prev_count) / (count - prev_count)
                        results[key] = prev_le + frac * (le_val - prev_le)
                    break
                prev_count = count
                prev_le = le_val if le_val != float("inf") else prev_le
            else:
                results[key] = prev_le

        return results

    # ------------------------------------------------------------------
    # Main collection
    # ------------------------------------------------------------------

    def collect(self) -> str:
        lines = []

        # ---- API Server ----
        # Primary: kubectl get --raw /metrics (uses kubeconfig auth)
        apiserver_raw = self._kubectl_raw("/metrics")
        if not apiserver_raw:
            # Fallback: raw HTTPS with bearer token
            apiserver_raw = self._https_get(
                f"{self.apiserver_url}/metrics",
                ca_cert=self.ca_cert,
                client_cert=self.client_cert,
                client_key=self.client_key,
                bearer_token=self.token,
            )
        apiserver_up = 1 if apiserver_raw else 0

        lines.append(f"# HELP {METRIC_PREFIX}_apiserver_up API server reachability (1=up)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_apiserver_up gauge\n")
        lines.append(f"{METRIC_PREFIX}_apiserver_up {apiserver_up}\n")

        if apiserver_raw:
            parsed_api = self._parse_prometheus_metrics(apiserver_raw)

            lines.append(f"# HELP {METRIC_PREFIX}_apiserver_request_rate Requests by verb and code\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_apiserver_request_rate gauge\n")
            verb_code_counts: Dict[str, Dict[str, float]] = {}
            for labels, value in parsed_api.get("apiserver_request_total", []):
                verb = labels.get("verb", "unknown")
                code = labels.get("code", "unknown")
                verb_code_counts.setdefault(verb, {})[code] = value
            for verb, codes in verb_code_counts.items():
                for code, val in codes.items():
                    lines.append(f'{METRIC_PREFIX}_apiserver_request_rate{{verb="{verb}",code="{code}"}} {val}\n')

            p99_latencies = self._histogram_quantile(parsed_api, "apiserver_request_duration_seconds", 0.99)
            lines.append(f"# HELP {METRIC_PREFIX}_apiserver_request_duration_p99_seconds P99 request latency\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_apiserver_request_duration_p99_seconds gauge\n")
            for verb, val in p99_latencies.items():
                if verb != "WATCH":
                    lines.append(f'{METRIC_PREFIX}_apiserver_request_duration_p99_seconds{{verb="{verb}"}} {val:.6f}\n')

            lines.append(f"# HELP {METRIC_PREFIX}_apiserver_storage_objects Stored objects by resource\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_apiserver_storage_objects gauge\n")
            for labels, value in parsed_api.get("apiserver_storage_objects", []):
                resource = labels.get("resource", "unknown")
                lines.append(f'{METRIC_PREFIX}_apiserver_storage_objects{{resource="{resource}"}} {value}\n')

        # ---- etcd ----
        # etcd requires mTLS client certs — use raw HTTPS with etcd-specific certs.
        # kubectl cannot proxy to etcd, so this is the only method.
        etcd_raw = None
        for port in [2381, 2379]:
            etcd_raw = self._https_get(
                f"https://127.0.0.1:{port}/metrics",
                ca_cert=self.etcd_ca,
                client_cert=self.etcd_cert,
                client_key=self.etcd_key,
            )
            if etcd_raw:
                logger.debug(f"etcd metrics collected from port {port}")
                break

        etcd_up = 1 if etcd_raw else 0
        lines.append(f"# HELP {METRIC_PREFIX}_etcd_up etcd reachability (1=up)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_etcd_up gauge\n")
        lines.append(f"{METRIC_PREFIX}_etcd_up {etcd_up}\n")

        if etcd_raw:
            parsed_etcd = self._parse_prometheus_metrics(etcd_raw)

            has_leader = 0.0
            for _, v in parsed_etcd.get("etcd_server_has_leader", []):
                has_leader = v
            lines.append(f"# HELP {METRIC_PREFIX}_etcd_has_leader etcd leader presence (1=has leader)\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_etcd_has_leader gauge\n")
            lines.append(f"{METRIC_PREFIX}_etcd_has_leader {has_leader}\n")

            leader_changes = 0.0
            for _, v in parsed_etcd.get("etcd_server_leader_changes_seen_total", []):
                leader_changes = v
            lines.append(f"# HELP {METRIC_PREFIX}_etcd_leader_changes_total Total etcd leader changes\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_etcd_leader_changes_total counter\n")
            lines.append(f"{METRIC_PREFIX}_etcd_leader_changes_total {leader_changes}\n")

            proposals_failed = 0.0
            for _, v in parsed_etcd.get("etcd_server_proposals_failed_total", []):
                proposals_failed += v
            lines.append(f"# HELP {METRIC_PREFIX}_etcd_proposals_failed_total Failed etcd proposals\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_etcd_proposals_failed_total counter\n")
            lines.append(f"{METRIC_PREFIX}_etcd_proposals_failed_total {proposals_failed}\n")

            wal_p99 = self._histogram_quantile(parsed_etcd, "etcd_disk_wal_fsync_duration_seconds", 0.99)
            p99_val = list(wal_p99.values())[0] if wal_p99 else 0.0
            lines.append(f"# HELP {METRIC_PREFIX}_etcd_wal_fsync_duration_p99_seconds P99 etcd WAL fsync\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_etcd_wal_fsync_duration_p99_seconds gauge\n")
            lines.append(f"{METRIC_PREFIX}_etcd_wal_fsync_duration_p99_seconds {p99_val:.6f}\n")

        # ---- Scheduler ----
        # kube-scheduler exposes /metrics on port 10259 via mTLS.
        # Pass ca_cert=None so SSL uses CERT_NONE (equivalent to curl -k) — the
        # scheduler's serving cert may use its own CA, not /etc/kubernetes/pki/ca.crt.
        # We still send the client cert for authentication; only server verification is skipped.
        sched_raw = self._https_get(
            "https://127.0.0.1:10259/metrics",
            ca_cert=None,
            client_cert=self.client_cert or f"{K8S_PKI_DIR}/apiserver-kubelet-client.crt",
            client_key=self.client_key or f"{K8S_PKI_DIR}/apiserver-kubelet-client.key",
        )
        if not sched_raw:
            # Fallback: proxy through apiserver
            sched_raw = self._kubectl_raw(
                "/api/v1/namespaces/kube-system/services/https:kube-scheduler:https/proxy/metrics"
            )

        sched_up = 1 if sched_raw else 0
        lines.append(f"# HELP {METRIC_PREFIX}_scheduler_up kube-scheduler reachability (1=up)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_up gauge\n")
        lines.append(f"{METRIC_PREFIX}_scheduler_up {sched_up}\n")

        if sched_raw:
            parsed_sched = self._parse_prometheus_metrics(sched_raw)

            lines.append(f"# HELP {METRIC_PREFIX}_scheduler_pending_pods Pending pods by queue\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_pending_pods gauge\n")
            for labels, value in parsed_sched.get("scheduler_pending_pods", []):
                queue = labels.get("queue", "unknown")
                lines.append(f'{METRIC_PREFIX}_scheduler_pending_pods{{queue="{queue}"}} {value}\n')

            lines.append(f"# HELP {METRIC_PREFIX}_scheduler_schedule_attempts_total Scheduling attempts\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_scheduler_schedule_attempts_total counter\n")
            for labels, value in parsed_sched.get("scheduler_schedule_attempts_total", []):
                result_label = labels.get("result", "unknown")
                lines.append(f'{METRIC_PREFIX}_scheduler_schedule_attempts_total{{result="{result_label}"}} {value}\n')

        # ---- Controller Manager ----
        # Same reasoning as scheduler: use ca_cert=None (CERT_NONE) to skip server
        # cert verification while still sending the client cert for mTLS auth.
        cm_raw = self._https_get(
            "https://127.0.0.1:10257/metrics",
            ca_cert=None,
            client_cert=self.client_cert or f"{K8S_PKI_DIR}/apiserver-kubelet-client.crt",
            client_key=self.client_key or f"{K8S_PKI_DIR}/apiserver-kubelet-client.key",
        )
        if not cm_raw:
            cm_raw = self._kubectl_raw(
                "/api/v1/namespaces/kube-system/services/https:kube-controller-manager:https/proxy/metrics"
            )

        cm_up = 1 if cm_raw else 0
        lines.append(f"# HELP {METRIC_PREFIX}_controller_manager_up controller-manager reachability (1=up)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_controller_manager_up gauge\n")
        lines.append(f"{METRIC_PREFIX}_controller_manager_up {cm_up}\n")

        if cm_raw:
            parsed_cm = self._parse_prometheus_metrics(cm_raw)

            lines.append(f"# HELP {METRIC_PREFIX}_controller_manager_workqueue_depth Work queue depth\n")
            lines.append(f"# TYPE {METRIC_PREFIX}_controller_manager_workqueue_depth gauge\n")
            for labels, value in parsed_cm.get("workqueue_depth", []):
                queue_name = labels.get("name", "unknown")
                lines.append(
                    f'{METRIC_PREFIX}_controller_manager_workqueue_depth{{queue_name="{queue_name}"}} {value}\n'
                )

        # ---- Node info via kubectl ----
        nodes_data = self._kubectl(["get", "nodes"])
        lines.append(f"# HELP {METRIC_PREFIX}_node_info Node info for table display (always 1)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_node_info gauge\n")
        if nodes_data and "items" in nodes_data:
            for item in nodes_data["items"]:
                node_name = item.get("metadata", {}).get("name", "unknown")
                node_labels = item.get("metadata", {}).get("labels", {})
                role = "control-plane" if any("control-plane" in k or "master" in k for k in node_labels) else "worker"
                conditions = item.get("status", {}).get("conditions", [])
                ready_cond = next((c for c in conditions if c.get("type") == "Ready"), {})
                status = "Ready" if ready_cond.get("status") == "True" else "NotReady"
                version = item.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", "unknown")
                lines.append(
                    f'{METRIC_PREFIX}_node_info{{node="{node_name}",role="{role}",'
                    f'status="{status}",version="{version}"}} 1\n'
                )

        # ---- Pending/Failed Pods via kubectl ----
        pods_data = self._kubectl(
            [
                "get",
                "pods",
                "--all-namespaces",
                "--field-selector=status.phase!=Running,status.phase!=Succeeded",
            ]
        )
        lines.append(f"# HELP {METRIC_PREFIX}_pod_info Pending or failed pod info (always 1)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_pod_info gauge\n")
        if pods_data and "items" in pods_data:
            for item in pods_data["items"]:
                pod_name = item.get("metadata", {}).get("name", "unknown")
                namespace = item.get("metadata", {}).get("namespace", "default")
                phase = item.get("status", {}).get("phase", "Unknown")
                node_nm = item.get("spec", {}).get("nodeName", "") or ""
                container_statuses = item.get("status", {}).get("containerStatuses", [])
                reason = ""
                if container_statuses:
                    waiting = container_statuses[0].get("state", {}).get("waiting", {})
                    reason = waiting.get("reason", "") or ""
                p = pod_name.replace('"', "'")
                ns = namespace.replace('"', "'")
                r = reason.replace('"', "'")
                nd = node_nm.replace('"', "'")
                lines.append(
                    f'{METRIC_PREFIX}_pod_info{{pod="{p}",namespace="{ns}",'
                    f'phase="{phase}",reason="{r}",node="{nd}"}} 1\n'
                )

        # ---- Component health via readyz ----
        # ca_cert=None for scheduler and CM so SSL uses CERT_NONE (skip server
        # cert verification, same as curl -k). Client cert is still sent for mTLS.
        component_checks = {
            "kube-apiserver": (f"{self.apiserver_url}/readyz", None, None, None, self.token),
            "kube-scheduler": (
                "https://127.0.0.1:10259/readyz",
                None,
                self.client_cert or f"{K8S_PKI_DIR}/apiserver-kubelet-client.crt",
                self.client_key or f"{K8S_PKI_DIR}/apiserver-kubelet-client.key",
                None,
            ),
            "kube-controller-manager": (
                "https://127.0.0.1:10257/readyz",
                None,
                self.client_cert or f"{K8S_PKI_DIR}/apiserver-kubelet-client.crt",
                self.client_key or f"{K8S_PKI_DIR}/apiserver-kubelet-client.key",
                None,
            ),
            "etcd": ("https://127.0.0.1:2379/health", self.etcd_ca, self.etcd_cert, self.etcd_key, None),
        }
        lines.append(f"# HELP {METRIC_PREFIX}_component_health Component health (1=healthy)\n")
        lines.append(f"# TYPE {METRIC_PREFIX}_component_health gauge\n")
        for component, (endpoint, ca, cc, ck, tok) in component_checks.items():
            # For apiserver, also try kubectl get --raw /readyz
            if component == "kube-apiserver":
                resp = self._kubectl_raw("/readyz")
                if not resp:
                    resp = self._https_get(endpoint, ca_cert=ca, client_cert=cc, client_key=ck, bearer_token=tok)
            else:
                resp = self._https_get(endpoint, ca_cert=ca, client_cert=cc, client_key=ck, bearer_token=tok)

            healthy = 0
            if resp:
                resp_lower = resp.lower().strip()
                if "ok" in resp_lower or "healthy" in resp_lower:
                    healthy = 1
            ep = endpoint.replace('"', "'")
            lines.append(f'{METRIC_PREFIX}_component_health{{component="{component}",endpoint="{ep}"}} {healthy}\n')

        return "".join(lines) if lines else "# no k8s metrics collected\n"


_collector: Optional[K8sMetricsCollector] = None


def collector_loop(interval: int):
    global metrics_output
    while True:
        try:
            if _collector:
                data = _collector.collect()
                with metrics_lock:
                    metrics_output = data
            logger.debug("K8s metrics collected")
        except Exception as e:
            logger.error(f"Collection error: {e}")
        time.sleep(interval)


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            with metrics_lock:
                data = metrics_output
            body = data.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes Control Plane Metrics Exporter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--kubeconfig", type=str, default=None)
    parser.add_argument("--apiserver-url", type=str, default="https://127.0.0.1:6443")
    # Generic TLS (for raw HTTPS fallback)
    parser.add_argument("--ca-cert", type=str, default=None)
    parser.add_argument("--client-cert", type=str, default=None)
    parser.add_argument("--client-key", type=str, default=None)
    parser.add_argument("--token-file", type=str, default=None)
    # etcd-specific TLS (auto-detected from /etc/kubernetes/pki/etcd/ if not set)
    parser.add_argument(
        "--etcd-ca-cert",
        type=str,
        default=None,
        help="etcd CA cert (default: auto-detect from /etc/kubernetes/pki/etcd/)",
    )
    parser.add_argument("--etcd-client-cert", type=str, default=None, help="etcd client cert (default: auto-detect)")
    parser.add_argument("--etcd-client-key", type=str, default=None, help="etcd client key (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect kubeconfig
    kubeconfig = args.kubeconfig
    if not kubeconfig:
        for candidate in ["/etc/kubernetes/admin.conf", os.path.expanduser("~/.kube/config")]:
            if os.path.exists(candidate):
                kubeconfig = candidate
                logger.info(f"Using kubeconfig: {kubeconfig}")
                break

    if not kubeconfig:
        logger.warning(
            "No kubeconfig found — kubectl-based metrics will fail. "
            "Pass --kubeconfig or ensure /etc/kubernetes/admin.conf exists."
        )

    _collector = K8sMetricsCollector(
        kubeconfig=kubeconfig,
        apiserver_url=args.apiserver_url,
        ca_cert=args.ca_cert,
        client_cert=args.client_cert,
        client_key=args.client_key,
        token_file=args.token_file,
        etcd_ca_cert=args.etcd_ca_cert,
        etcd_client_cert=args.etcd_client_cert,
        etcd_client_key=args.etcd_client_key,
    )

    logger.info(f"Starting K8s Control Plane exporter on port {args.port}, interval {args.interval}s")

    t = threading.Thread(target=collector_loop, args=(args.interval,), daemon=True)
    t.start()
    time.sleep(3)

    server = http.server.HTTPServer(("", args.port), MetricsHandler)
    logger.info(f"K8s Control Plane exporter listening on :{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()
