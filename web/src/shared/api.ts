// Minimal fetch wrapper for the CVS API. All tiles share one origin (the Go
// daemon), so paths are relative.

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

async function errorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { error?: string };
    return body?.error ?? "";
  } catch {
    return "";
  }
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `POST ${path} failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(path, { method: "DELETE", headers: { Accept: "application/json" } });
  if (!res.ok && res.status !== 204) {
    throw new Error((await errorDetail(res)) || `DELETE ${path} failed: ${res.status}`);
  }
}

// One CVS test suite, as reported by `cvs list --json`.
export interface Suite {
  name: string;
  module_path: string;
  module: string;
  package: string;
  category: string;
}

export async function listSuites(): Promise<Suite[]> {
  const data = await apiGet<{ suites: Suite[] }>("/api/v1/suites");
  return data.suites ?? [];
}

// A node of the inferred config form (from GET /suites/{suite}/schema).
export interface Field {
  name: string;
  type: "object" | "array" | "string" | "number" | "boolean" | "null";
  description?: string;
  example?: string;
  fields?: Field[];
  item?: Field;
}

export interface SuiteSchema {
  suite: string;
  test_name: string;
  has_config: boolean;
  config_path: string | null;
  format?: string;
  schema: Field[];
  example: unknown;
}

export async function getSuiteSchema(suite: string, example?: string): Promise<SuiteSchema> {
  const q = example ? `?example=${encodeURIComponent(example)}` : "";
  return apiGet<SuiteSchema>(`/api/v1/suites/${encodeURIComponent(suite)}/schema${q}`);
}

// One example config file, with facets parsed from its
// category/framework/{platform}_{framework}_{model}_{topology} path. Facet
// fields are empty for flat single-config suites.
export interface ConfigRef {
  name: string;
  path: string;
  format: string;
  category?: string;
  framework?: string;
  platform?: string;
  model?: string;
  topology?: string;
}

export async function getSuiteExamples(suite: string): Promise<ConfigRef[]> {
  const data = await apiGet<{ examples: ConfigRef[] }>(
    `/api/v1/suites/${encodeURIComponent(suite)}/examples`,
  );
  return data.examples ?? [];
}

export interface ConfigFacets {
  category: string[];
  framework: string[];
  platform: string[];
  model: string[];
  topology: string[];
}

export interface ConfigCatalog {
  configs: ConfigRef[];
  facets: ConfigFacets;
}

export async function getConfigCatalog(): Promise<ConfigCatalog> {
  return apiGet<ConfigCatalog>("/api/v1/suites/catalog");
}

// --- Inventory (the inventory-first gate shared by all tiles) ---

export type AuthMethod = "key" | "password";

export interface JumpHost {
  host: string;
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  node_username?: string;
  node_key_file?: string;
}

export interface NodeStatus {
  host: string;
  reachable: boolean;
  gpu_type?: string;
  gpu_count?: number;
  rocm_version?: string;
  error?: string;
  checked_at?: string;
}

export interface Inventory {
  nodes: string[];
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  jump_host?: JumpHost | null;
  statuses?: NodeStatus[];
  updated_at: string;
}

export interface InventoryStatus {
  configured: boolean;
  inventory?: Inventory;
}

export interface SaveInventoryRequest {
  nodes: string[];
  username: string;
  auth_method: AuthMethod;
  key_name?: string;
  jump_host?: JumpHost | null;
}

export async function getInventory(): Promise<InventoryStatus> {
  return apiGet<InventoryStatus>("/api/v1/inventory");
}

export async function saveInventory(req: SaveInventoryRequest): Promise<InventoryStatus> {
  return apiPost<InventoryStatus>("/api/v1/inventory", req);
}

/** Removes saved fleet inventory. Uploaded SSH keys are not deleted. */
export async function deleteInventory(): Promise<void> {
  await apiDelete("/api/v1/inventory");
}

// Runs the F2 connectivity + basic-info sweep and returns the updated inventory
// (with per-node reachability, GPU type/count, and ROCm version).
export async function probeInventory(): Promise<InventoryStatus> {
  return apiPost<InventoryStatus>("/api/v1/inventory/probe", {});
}

export async function listKeys(): Promise<string[]> {
  const data = await apiGet<{ keys: string[] }>("/api/v1/inventory/keys");
  return data.keys ?? [];
}

// --- Saved-cluster catalog (S3b) ---

export interface Cluster {
  id: string;
  name: string;
  nodes: string[];
  head_node?: string;
  file_path: string;
  source: string;
  created_at: string;
  updated_at: string;
}

export async function listClusters(): Promise<Cluster[]> {
  const data = await apiGet<{ clusters: Cluster[] }>("/api/v1/clusters");
  return data.clusters ?? [];
}

export async function createCluster(req: {
  name: string;
  nodes: string[];
  head_node?: string;
}): Promise<Cluster> {
  return apiPost<Cluster>("/api/v1/clusters", req);
}

export async function updateCluster(
  id: string,
  req: { name: string; nodes: string[]; head_node?: string },
): Promise<Cluster> {
  const res = await fetch(`/api/v1/clusters/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `PUT cluster failed: ${res.status}`);
  }
  return (await res.json()) as Cluster;
}

export async function deleteCluster(id: string): Promise<void> {
  return apiDelete(`/api/v1/clusters/${encodeURIComponent(id)}`);
}

// Fetches the raw generated cluster_json file for a saved cluster.
export async function getClusterContent(id: string): Promise<string> {
  const res = await fetch(`/api/v1/clusters/${encodeURIComponent(id)}/content`);
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `cluster content failed: ${res.status}`);
  }
  return res.text();
}

// --- Test execution (S3) ---

export type ExecutionStatus =
  | "queued"
  | "running"
  | "passed"
  | "failed"
  | "error"
  | "interrupted";

export interface Execution {
  id: string;
  suite: string;
  cluster_id: string;
  cluster_file: string;
  config_path?: string;
  status: ExecutionStatus;
  exit_code?: number;
  error?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  // Derived server-side: whether a pytest HTML report was produced.
  has_report?: boolean;
}

export function isTerminal(s: ExecutionStatus): boolean {
  return s === "passed" || s === "failed" || s === "error" || s === "interrupted";
}

export async function executeTest(req: {
  suite: string;
  cluster_id: string;
  config?: unknown;
}): Promise<Execution> {
  return apiPost<Execution>("/api/v1/tests/execute", req);
}

// Launches a fresh execution from an existing one, reusing its suite, cluster,
// and staged config. The cluster is re-resolved server-side at rerun time.
export async function rerunExecution(id: string): Promise<Execution> {
  return apiPost<Execution>(`/api/v1/executions/${encodeURIComponent(id)}/rerun`, {});
}

export async function listExecutions(): Promise<Execution[]> {
  const data = await apiGet<{ executions: Execution[] }>("/api/v1/executions");
  return data.executions ?? [];
}

export async function getExecution(id: string): Promise<Execution> {
  return apiGet<Execution>(`/api/v1/executions/${encodeURIComponent(id)}`);
}

// URL of the pytest HTML report for an execution (opens directly in a new tab;
// its per-test log links resolve against the same artifacts path).
export function executionReportUrl(id: string): string {
  return `/api/v1/executions/${encodeURIComponent(id)}/artifacts/report.html`;
}

// URL of the raw combined run log file for an execution (opens as plain text).
export function executionLogUrl(id: string): string {
  return `/api/v1/executions/${encodeURIComponent(id)}/artifacts/output.log`;
}

// Fetches the generated cluster_file (cluster_json) used for a run, or null if
// it is no longer available.
export async function getExecutionClusterFile(id: string): Promise<string | null> {
  const res = await fetch(`/api/v1/executions/${encodeURIComponent(id)}/cluster`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error((await errorDetail(res)) || `cluster file failed: ${res.status}`);
  return res.text();
}

// Fetches the staged config.json used for a run, or null when the run had no
// config (flat suites) or the file is gone.
export async function getExecutionConfig(id: string): Promise<string | null> {
  const res = await fetch(`/api/v1/executions/${encodeURIComponent(id)}/artifacts/config.json`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error((await errorDetail(res)) || `config file failed: ${res.status}`);
  return res.text();
}

export async function getExecutionLogs(
  id: string,
): Promise<{ id: string; status: ExecutionStatus; logs: string }> {
  return apiGet(`/api/v1/executions/${encodeURIComponent(id)}/logs`);
}

// --- Live streaming over WebSocket (S4) ---

// Envelope shared by all WS streams: {type, execution_id, timestamp, data}.
export interface WSMessage {
  type: "log" | "status" | "error" | "completion" | "metrics";
  execution_id?: string;
  timestamp: string;
  data?: unknown;
}

// wsURL builds an absolute ws(s):// URL from a same-origin path.
function wsURL(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

export interface ExecutionStreamHandlers {
  onLog: (line: string) => void;
  onStatus: (ex: Execution) => void;
  // onClose fires when the socket closes for any reason (normal seal on
  // completion, error, or a dropped subscriber). Callers should reconcile the
  // authoritative status/logs via REST here, since live frames are best-effort.
  onClose?: () => void;
  onError?: (err: Event) => void;
}

// streamExecution opens a WS that replays the execution's backlog then streams
// live log lines and status frames. Returns a close() function. The server seals
// the connection once the execution reaches a terminal status.
//
// Live frames are best-effort: under a heavy log burst the server may drop a
// slow subscriber rather than block the executor, so the terminal status frame
// can be missed. Always reconcile via onClose against the REST store.
export function streamExecution(id: string, h: ExecutionStreamHandlers): () => void {
  const ws = new WebSocket(wsURL(`/ws/executions/${encodeURIComponent(id)}`));
  ws.onmessage = (ev) => {
    let msg: WSMessage;
    try {
      msg = JSON.parse(ev.data as string) as WSMessage;
    } catch {
      return;
    }
    if (msg.type === "log") {
      const line = (msg.data as { line?: string } | undefined)?.line;
      if (line != null) h.onLog(line);
    } else if (msg.type === "status" || msg.type === "completion") {
      h.onStatus(msg.data as Execution);
    }
  };
  if (h.onClose) ws.onclose = () => h.onClose!();
  if (h.onError) ws.onerror = h.onError;
  return () => {
    ws.onclose = null; // caller-initiated close should not trigger reconcile
    try {
      ws.close();
    } catch {
      // ignore
    }
  };
}

// --- Cluster Monitor (S6) ---

// One node's row in the Cluster Monitor grid: identity + last probe result.
export interface ClusterNode {
  host: string;
  probed: boolean;
  reachable: boolean;
  gpu_type?: string;
  gpu_count?: number;
  rocm_version?: string;
  error?: string;
  checked_at?: string;
}

export interface ClusterNodes {
  nodes: ClusterNode[];
  total: number;
  configured: boolean;
  updated_at?: string;
}

export async function listClusterNodes(): Promise<ClusterNodes> {
  return apiGet<ClusterNodes>("/api/v1/clustermon/nodes");
}

// --- Cluster Monitor live GPU metrics (S7) ---

export interface PCIeInfo {
  width: string;
  speed: string;
  bandwidth: string;
  replay_count: number;
  l0_to_recovery_count: number;
  nak_sent_count: number;
  nak_received_count: number;
}

export interface ECCInfo {
  correctable: number;
  uncorrectable: number;
  deferred: number;
}

export interface XGMIInfo {
  status?: string;
  error_count: number;
}

export interface GPUMetric {
  index: number;
  utilization_pct: number;
  mem_total_mb: number;
  mem_used_mb: number;
  mem_used_pct: number;
  temp_edge_c: number;
  temp_hotspot_c: number;
  temp_mem_c: number;
  power_w: number;
  pcie?: PCIeInfo;
  ecc?: ECCInfo;
  xgmi?: XGMIInfo;
}

export interface NodeGPUMetrics {
  host: string;
  gpus: GPUMetric[] | null;
  error?: string;
}

// --- Cluster Monitor live NIC metrics (S9) ---

export interface RDMALink {
  device: string;
  state: string;
  physical_state: string;
  netdev: string;
}

export interface RDMADeviceStats {
  device: string;
  stats: Record<string, number>;
}

export interface RDMAResource {
  device: string;
  values: Record<string, number>;
}

export interface NICInterface {
  name: string;
  mtu?: string;
  state?: string;
  mac_addr?: string;
  ipv4?: string[];
  ipv6?: string[];
  flags?: string;
}

export interface EthStats {
  iface: string;
  stats: Record<string, number>;
}

export interface NodeNICMetrics {
  host: string;
  rdma_links?: RDMALink[];
  rdma_stats?: RDMADeviceStats[];
  rdma_resources?: RDMAResource[];
  interfaces?: NICInterface[];
  eth_stats?: EthStats[];
  error?: string;
}

// MetricsSnapshot is one live sweep: GPU + NIC collectors for the fleet (S9).
export interface MetricsSnapshot {
  collected_at: string | null;
  gpu: NodeGPUMetrics[];
  nic: NodeNICMetrics[];
}

// --- Cluster Monitor on-demand software collectors (S9b, TTL-cached) ---

export interface GPUSoftwareInfo {
  amdsmi_tool: string;
  amdsmi_library: string;
  rocm_version: string;
  amdgpu_version: string;
  amd_hsmp_version: string;
}

export interface FirmwareEntry {
  fw_id: string;
  fw_version: string;
}

export interface GPUFirmware {
  gpu: number;
  fw_list: FirmwareEntry[];
}

export interface NodeGPUSoftware {
  host: string;
  version?: GPUSoftwareInfo;
  firmware?: GPUFirmware[];
  error?: string;
}

export interface GPUSoftwareSnapshot {
  collected_at: string;
  nodes: NodeGPUSoftware[];
}

export interface DevlinkDevice {
  pci_address: string;
  driver: string;
  vendor: string;
  serial_number: string;
  board_serial: string;
  board_id: string;
  asic_id: string;
  asic_rev: string;
  fw_version: string;
  fw_psid: string;
  fw_mgmt: string;
  fw_mgmt_api: string;
  fw_cpld: string;
  fw_heartbeat: string;
}

export interface NodeDevlink {
  host: string;
  devices?: DevlinkDevice[];
  error?: string;
}

export interface DevlinkSnapshot {
  collected_at: string;
  nodes: NodeDevlink[];
  collecting?: boolean;
}

export interface GPUSoftwareSnapshotOrPending extends GPUSoftwareSnapshot {
  collecting?: boolean;
}

export async function getGpuSoftware(): Promise<GPUSoftwareSnapshotOrPending> {
  return apiGet<GPUSoftwareSnapshotOrPending>("/api/v1/clustermon/software/gpu");
}

export async function getNicDevlink(): Promise<DevlinkSnapshot> {
  return apiGet<DevlinkSnapshot>("/api/v1/clustermon/software/nic/devlink");
}

// --- Cluster Monitor logs (S9c) ---

export interface NodeLogs {
  host: string;
  amd_logs: string;
  dmesg_errors: string;
  userspace_errors: string;
  error?: string;
}

export interface LogsSnapshot {
  collected_at: string;
  nodes: NodeLogs[];
  collecting?: boolean;  // true when a background re-sweep is in progress
}

export interface SearchResult {
  host: string;
  output: string;
}

export interface SearchResponse {
  grep_command: string;
  results: SearchResult[];
  total_nodes_searched: number;
  nodes_with_results: number;
}

// Returns null when the first collection is still running (HTTP 204).
export async function getDmesgLogs(): Promise<LogsSnapshot | null> {
  const res = await fetch("/api/v1/clustermon/logs/dmesg", {
    headers: { Accept: "application/json" },
  });
  if (res.status === 204) return null;
  if (!res.ok) throw new Error(`GET /api/v1/clustermon/logs/dmesg failed: ${res.status}`);
  return (await res.json()) as LogsSnapshot;
}

export async function searchLogs(grepCommand: string): Promise<SearchResponse> {
  const res = await fetch(
    `/api/v1/clustermon/logs/search?grep_command=${encodeURIComponent(grepCommand)}`,
    { headers: { Accept: "application/json" } },
  );
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `log search failed: ${res.status}`);
  }
  return (await res.json()) as SearchResponse;
}

export async function getLatestMetrics(): Promise<MetricsSnapshot> {
  return apiGet<MetricsSnapshot>("/api/v1/clustermon/metrics/latest");
}

// --- Cluster Monitor Dashboard (S9e) ---

export interface ClusterStatus {
  total_nodes: number;
  healthy_nodes: number;
  unhealthy_nodes: number;
  unreachable_nodes: number;
  total_gpus: number;
  avg_gpu_util_pct: number;
  avg_gpu_mem_pct: number;
  avg_gpu_temp_c: number;
  collected_at: string | null;
  metrics_available: boolean;
}

export interface NodeDetail {
  host: string;
  gpus: GPUMetric[];
  rdma_links: RDMALink[];
  error?: string;
}

export async function getClusterStatus(): Promise<ClusterStatus> {
  return apiGet<ClusterStatus>("/api/v1/clustermon/cluster/status");
}

export async function getNodeDetail(host: string): Promise<NodeDetail> {
  return apiGet<NodeDetail>(`/api/v1/clustermon/nodes/${encodeURIComponent(host)}`);
}

// Triggers an on-demand GPU metrics sweep over the reachable nodes and returns
// the fresh snapshot.
export async function collectMetrics(): Promise<MetricsSnapshot> {
  return apiPost<MetricsSnapshot>("/api/v1/clustermon/metrics/collect", {});
}

// --- Cluster Monitor live metrics over WebSocket (S8) ---

// streamClusterMetrics subscribes to the server-side metrics broadcast. The
// server sends the cached latest snapshot on connect, then pushes each snapshot
// the poll loop collects (default every 60s). Returns a close() function.
export function streamClusterMetrics(
  onSnapshot: (snap: MetricsSnapshot) => void,
  onClose?: () => void,
): () => void {
  const ws = new WebSocket(wsURL("/ws/clustermon"));
  ws.onmessage = (ev) => {
    let msg: WSMessage & { data?: MetricsSnapshot };
    try {
      msg = JSON.parse(ev.data as string);
    } catch {
      return;
    }
    if (msg.type === "metrics" && msg.data) onSnapshot(msg.data);
  };
  if (onClose) ws.onclose = () => onClose();
  return () => {
    ws.onclose = null;
    try {
      ws.close();
    } catch {
      // ignore
    }
  };
}

export async function uploadSshKey(file: File): Promise<{ key_name: string; path: string }> {
  const form = new FormData();
  form.append("key", file);
  const res = await fetch("/api/v1/inventory/keys", { method: "POST", body: form });
  if (!res.ok) {
    throw new Error((await errorDetail(res)) || `key upload failed: ${res.status}`);
  }
  return (await res.json()) as { key_name: string; path: string };
}

// --- S9g Network Topology (LLDP) ---

export interface LLDPNeighbor {
  interface: string;
  chassis: string;
  chassis_id?: string;
  port: string;
  description?: string;
  mgmt_ip?: string;
}

export interface NodeLLDPData {
  host: string;
  neighbors?: LLDPNeighbor[];
  error?: string;
}

export interface TopologySnapshot {
  collected_at: string;
  nodes: NodeLLDPData[];
}

export async function getTopologyLLDP(): Promise<TopologySnapshot> {
  return apiGet<TopologySnapshot>("/api/v1/clustermon/topology/lldp");
}

// --- Probe status ---

export interface ProbeStatus {
  ready: boolean;
  reachable: number;
  unreachable: number;
  total: number;
}

export async function getProbeStatus(): Promise<ProbeStatus> {
  return apiGet<ProbeStatus>("/api/v1/inventory/probe-status");
}
