"""AMD GPU node-health checks with optional MI4XX fabric admission.

The check intentionally validates state only.  It never loads a driver, starts
an AIFM agent, changes a vPOD, or masks an IFoE station.  Those are platform
recovery actions and must happen before CVS admits a node to a benchmark.

AFM and AMD-SMI are requested in JSON mode.  The parser is semantic rather
than tied to a single package-version schema so that the same check can cover
MI4XX rack images with compatible AFM and AMD-SMI interfaces.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cvs.lib.preflight.base import PreflightCheck
from cvs.lib.preflight.ifoe_l2_connectivity import AfmctlPortParser, parse_afmctl_show_device_json


_BDF_KEYS = ("bdf", "device_bdf", "pci_bdf")
_GPU_ID_KEYS = ("gpu_id", "gpu", "id", "index")
_BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]$")

_KERNEL_FAILURE_RE = re.compile(
    r"(?:MES\([^)]*\).*SET_HW_RSRC|hw_init.*failed|Fatal error during GPU init|"
    r"fw_status\s*0xffffffff|memory training failed|hbm .*test failed|"
    r"(?:wafl|xgmi).*link training failed|discovery failed|probe with driver amdgpu failed)",
    re.IGNORECASE,
)


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalize_bdf(value: Any) -> Optional[str]:
    candidate = str(value or "").strip()
    return candidate.lower() if _BDF_RE.fullmatch(candidate) else None


def _normalise_state(value: Any) -> str:
    return str(value or "").strip().upper().replace("_", "-")


def _normalise_virtualization_mode(value: Any) -> str:
    """Canonicalise the AFM bare-metal spelling without accepting other modes."""
    state = _normalise_state(value)
    return "BARE-METAL" if state in {"BAREMETAL", "BARE-METAL", "BARE METAL"} else state


def _flatten_values(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten JSON dictionaries while retaining both leaf and path keys."""
    flattened: Dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return flattened
    for key, child in value.items():
        normalised = _normalise_key(key)
        path = f"{prefix}_{normalised}" if prefix else normalised
        if isinstance(child, Mapping):
            flattened.update(_flatten_values(child, path))
        else:
            flattened.setdefault(normalised, child)
            flattened[path] = child
    return flattened


def _first_value(flattened: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        normalised = _normalise_key(key)
        if normalised in flattened:
            return flattened[normalised]
    return None


def _walk_dicts(value: Any, inherited_bdf: Optional[str] = None) -> Iterable[Tuple[Dict[str, Any], Optional[str]]]:
    """Yield dictionaries and a BDF inherited from a BDF-keyed JSON envelope."""
    if isinstance(value, Mapping):
        direct_bdf = _first_value(_flatten_values(value), _BDF_KEYS)
        current_bdf = _normalize_bdf(direct_bdf) or inherited_bdf
        yield dict(value), current_bdf
        for key, child in value.items():
            key_bdf = _normalize_bdf(key)
            yield from _walk_dicts(child, key_bdf or current_bdf)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child, inherited_bdf)


def _parse_json(output: str, command_name: str) -> Tuple[Optional[Any], List[str]]:
    raw = (output or "").strip()
    if not raw:
        return None, [f"{command_name} returned empty output"]
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as exc:
        return None, [f"{command_name} did not return valid JSON: {exc.msg}"]


def parse_afmctl_device_json(output: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Adapt the shared AFM parser to the MI4XX admission policy."""
    if not (output or "").strip():
        return [], ["afmctl show device --json returned empty output"]
    devices, errors = parse_afmctl_show_device_json(output)
    for device in devices:
        device["config_phase"] = _normalise_state(device.get("config_phase"))
        device["virtualization_mode"] = _normalise_virtualization_mode(device.get("virtualization_mode"))
    return devices, errors


def parse_amd_smi_gpu_json(output: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return a de-duplicated GPU inventory from ``amd-smi list --json``."""
    document, errors = _parse_json(output, "amd-smi list --json")
    if errors:
        return [], errors

    gpus: List[Dict[str, Any]] = []
    seen = set()
    for record, inherited_bdf in _walk_dicts(document):
        flattened = _flatten_values(record)
        bdf = _normalize_bdf(_first_value(flattened, _BDF_KEYS)) or inherited_bdf
        gpu_id = _first_value(flattened, _GPU_ID_KEYS)
        if bdf is None and gpu_id is None:
            continue
        # AFM/IFoE BDFs are function .1; AMD-SMI lists GPU function .0.
        if bdf and not bdf.endswith(".0"):
            continue
        identity = bdf or f"gpu:{gpu_id}"
        if identity in seen:
            continue
        seen.add(identity)
        gpus.append({"bdf": bdf, "gpu_id": gpu_id})
    if not gpus:
        errors.append("amd-smi list JSON contained no GPU descriptors")
    return gpus, errors


def parse_station_masks(output: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse ``GPU-BDF station-mask`` lines emitted by :meth:`build_station_mask_command`."""
    masks: Dict[str, str] = {}
    errors: List[str] = []
    for raw in (output or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            errors.append(f"Malformed station-mask record {line!r}")
            continue
        gpu_bdf = _normalize_bdf(fields[0])
        if not gpu_bdf or not gpu_bdf.endswith(".0"):
            errors.append(f"Malformed GPU BDF in station-mask record {line!r}")
            continue
        mask = fields[1].lower()
        if not re.fullmatch(r"[0-9a-f]{18}", mask):
            errors.append(f"{gpu_bdf}: expected an 18-nibble station bitmap, got {mask!r}")
            continue
        if gpu_bdf in masks and masks[gpu_bdf] != mask:
            errors.append(f"{gpu_bdf}: conflicting station masks")
            continue
        masks[gpu_bdf] = mask
    if not masks and not errors:
        errors.append("station-mask command returned no records")
    return masks, errors


class NodeHealthCheck(PreflightCheck):
    """Read-only GPU node-health gate with optional MI4XX AFM validation.

    The generic phase validates AMDGPU/KFD readiness, GPU visibility, and
    current-boot kernel health.  ``fabric_checks=True`` adds the MI4XX
    AIFM/AFM/vPOD/station admission phase.
    """

    def __init__(
        self,
        phdl,
        *,
        expected_gpus_per_node: int = 4,
        fabric_checks: bool = True,
        expected_network_ports_per_device: int = 36,
        afmctl_path: str = "afmctl",
        amd_smi_path: str = "amd-smi",
        json_args: Optional[Sequence[str]] = None,
        use_sudo: bool = True,
        required_ifoe_modules: Optional[Sequence[str]] = None,
        agent_process_name: str = "inb-node-agent",
        agent_slot_ids: Optional[Mapping[str, int]] = None,
        readiness_timeout_seconds: int = 600,
        poll_interval_seconds: int = 15,
        allow_disabled_stations: bool = True,
        reject_partial_stations: bool = True,
        min_up_ports_per_gpu: int = 0,
        require_uniform_station_mask: bool = False,
        expected_station_masks: Optional[Mapping[str, str]] = None,
        expected_virtualization_mode: str = "bare-metal",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(phdl, config_dict)
        self.expected_gpus_per_node = int(expected_gpus_per_node)
        if self.expected_gpus_per_node < 1:
            raise ValueError("expected_gpus_per_node must be at least 1")
        self.fabric_checks = bool(fabric_checks)
        # MI4XX exposes one AFM/IFoE device for each GPU.  Keep this invariant
        # internal instead of asking customers for a duplicate device count.
        self.expected_ifoe_devices_per_node = self.expected_gpus_per_node
        self.expected_network_ports_per_device = int(expected_network_ports_per_device)
        self.afmctl_path = afmctl_path
        self.amd_smi_path = amd_smi_path
        raw_json_args = json_args or ["--json"]
        self.json_args = [str(raw_json_args)] if isinstance(raw_json_args, str) else [str(arg) for arg in raw_json_args]
        if not any(arg in ("--json", "-j") for arg in self.json_args):
            raise ValueError("json_args must request JSON output")
        self.use_sudo = bool(use_sudo)
        raw_modules = required_ifoe_modules or ["ifoe"]
        self.required_ifoe_modules = (
            [str(raw_modules)] if isinstance(raw_modules, str) else [str(m) for m in raw_modules]
        )
        self.agent_process_name = agent_process_name
        self.agent_slot_ids = {str(k): int(v) for k, v in (agent_slot_ids or {}).items()}
        self.readiness_timeout_seconds = max(0, int(readiness_timeout_seconds))
        self.poll_interval_seconds = max(1, int(poll_interval_seconds))
        self.allow_disabled_stations = bool(allow_disabled_stations)
        self.reject_partial_stations = bool(reject_partial_stations)
        self.min_up_ports_per_gpu = max(0, int(min_up_ports_per_gpu))
        self.require_uniform_station_mask = bool(require_uniform_station_mask)
        self.expected_station_masks = {}
        for raw_bdf, raw_mask in (expected_station_masks or {}).items():
            bdf = _normalize_bdf(raw_bdf)
            if not bdf:
                raise ValueError(f"Invalid expected station-mask BDF {raw_bdf!r}")
            mask = str(raw_mask).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{18}", mask):
                raise ValueError(f"{raw_bdf}: expected station mask must contain exactly 18 hexadecimal nibbles")
            self.expected_station_masks[bdf] = mask
        self.expected_virtualization_mode = _normalise_virtualization_mode(expected_virtualization_mode)

    def _privileged_parts(self, executable: str) -> List[str]:
        return (["sudo", "-n"] if self.use_sudo else []) + [executable]

    def build_afm_device_command(self) -> str:
        return " ".join(
            shlex.quote(part) for part in self._privileged_parts(self.afmctl_path) + ["show", "device", *self.json_args]
        )

    def build_afm_ports_command(self, bdfs: Sequence[str]) -> str:
        """Build one AFM port-inventory command for all local devices.

        ``afmctl show port`` accepts a comma-separated BDF list.  Gathering a
        node's complete local inventory in one command avoids an O(nodes^2)
        sequence of targeted SSH waves on a scale-up rack.
        """
        valid_bdfs = []
        for raw_bdf in bdfs:
            bdf = _normalize_bdf(raw_bdf)
            if not bdf:
                raise ValueError(f"Invalid AFM port-inventory BDF {raw_bdf!r}")
            valid_bdfs.append(bdf)
        if not valid_bdfs:
            raise ValueError("At least one AFM port-inventory BDF is required")
        return " ".join(
            shlex.quote(part)
            for part in self._privileged_parts(self.afmctl_path)
            + ["show", "port", "-b", ",".join(valid_bdfs), *self.json_args]
        )

    def build_gpu_inventory_command(self) -> str:
        executable = shlex.quote(self.amd_smi_path)
        arguments = " ".join(shlex.quote(part) for part in ["list", *self.json_args])
        privilege_prefix = "sudo -n " if self.use_sudo else ""
        return (
            f'_cvs_amd_smi="$(command -v {executable})" || '
            f'{{ printf \'%s\\n\' {shlex.quote(f"Unable to find {self.amd_smi_path} in PATH")} >&2; exit 127; }}; '
            'case "$_cvs_amd_smi" in /*) ;; *) '
            "printf '%s\\n' 'Resolved amd-smi path is not absolute' >&2; exit 127 ;; esac; "
            f'{privilege_prefix}"$_cvs_amd_smi" {arguments}'
        )

    def build_station_mask_command(self) -> str:
        mask_reader = "sudo -n cat" if self.use_sudo else "cat"
        return (
            "for _cvs_card in /sys/class/drm/card[0-9]*; do "
            "[ -e \"$_cvs_card/device/ualink/stations/lane_en_bitmap\" ] || continue; "
            "_cvs_bdf=$(basename \"$(readlink -f \"$_cvs_card/device\")\"); "
            f"_cvs_mask=$({mask_reader} \"$_cvs_card/device/ualink/stations/lane_en_bitmap\"); "
            "printf '%s %s\\n' \"$_cvs_bdf\" \"$_cvs_mask\"; "
            "done"
        )

    def _exec_all(self, command: str) -> Dict[str, str]:
        try:
            result = self.phdl.exec(command, timeout=90, print_console=False)
        except TypeError:
            result = self.phdl.exec(command, timeout=90)
        return result if isinstance(result, dict) else {}

    def _exec_commands_by_host(self, commands: Mapping[str, str]) -> Dict[str, str]:
        """Run one command per reachable host in one sharded SSH operation."""
        hosts = list(getattr(self.phdl, "reachable_hosts", []) or [])
        executor = getattr(self.phdl, "exec_cmd_list", None)
        if callable(executor):
            try:
                output = executor([commands.get(host, "true") for host in hosts], timeout=90, print_console=False)
                if isinstance(output, dict):
                    normalized = {}
                    for host in hosts:
                        value = output.get(host, "")
                        if isinstance(value, Mapping):
                            value = value.get("output", "")
                        normalized[host] = str(value)
                    return normalized
            except TypeError:
                pass
        # Test doubles and legacy SSH adapters may lack exec_cmd_list.  The
        # fallback remains functionally correct, but normal CVS Pssh objects
        # take the single-wave path above.
        return {host: str(self._exec_all(commands.get(host, "true")).get(host, "")) for host in hosts}

    @staticmethod
    def _module_command(module: str) -> str:
        return f"test -d /sys/module/{shlex.quote(module)} && printf 1 || printf 0"

    @staticmethod
    def _agent_command(process_name: str) -> str:
        # ``pgrep -af`` sees the remote shell command line. Match the first
        # character through a character class so the command itself cannot
        # satisfy its own process-name probe (for example ``[i]nb-node-agent``
        # matches the agent but not the literal probe command).
        if not process_name:
            return "false"
        regex = f"[{re.escape(process_name[0])}]{re.escape(process_name[1:])}"
        return f"pgrep -af -- {shlex.quote(regex)} || true"

    def _kernel_command(self) -> str:
        reader = "journalctl -k -b --no-pager"
        if self.use_sudo:
            reader = "sudo -n " + reader
        return f"{reader} 2>&1 || true"

    @staticmethod
    def _add_error(result: Dict[str, Any], message: str) -> None:
        result["errors"].append(message)
        result["status"] = "FAIL"

    def _evaluate_host_prerequisites(self, host_outputs: Dict[str, Dict[str, str]]) -> None:
        for node, result in self.results.items():
            probes = host_outputs.get(node, {})
            if probes.get("amdgpu", "").strip() != "1":
                self._add_error(result, "amdgpu kernel module is not loaded")
            if probes.get("kfd", "").strip() != "1":
                self._add_error(result, "/dev/kfd is absent")
            if self.fabric_checks:
                for module in self.required_ifoe_modules:
                    if probes.get(f"module:{module}", "").strip() != "1":
                        self._add_error(result, f"IFoE kernel module {module!r} is not loaded")
                agent_output = probes.get("agent", "")
                matching_lines = [line for line in agent_output.splitlines() if self.agent_process_name in line]
                if not matching_lines:
                    self._add_error(result, f"AIFM node agent {self.agent_process_name!r} is not running")
                else:
                    slot = self.agent_slot_ids.get(node)
                    if slot is not None and not any(
                        re.search(rf"(?:--?slot[-_]id[= ]|--?slot[= ]){re.escape(str(slot))}(?:\s|$)", line)
                        for line in matching_lines
                    ):
                        self._add_error(result, f"AIFM node agent is not running with required slot-id {slot}")
            kernel_output = probes.get("kernel", "")
            kernel_failures = [line.strip() for line in kernel_output.splitlines() if _KERNEL_FAILURE_RE.search(line)]
            result["kernel_failures"] = kernel_failures
            if kernel_failures:
                self._add_error(result, f"Current-boot amdgpu initialization failures: {kernel_failures[0]}")

    def _evaluate_gpu_inventory(self, outputs: Dict[str, str]) -> None:
        for node, result in self.results.items():
            gpus, errors = parse_amd_smi_gpu_json(outputs.get(node, ""))
            result["gpu_inventory"] = gpus
            result["raw_gpu_inventory"] = outputs.get(node, "")
            for error in errors:
                self._add_error(result, error)
            if len(gpus) != self.expected_gpus_per_node:
                self._add_error(
                    result,
                    f"Expected {self.expected_gpus_per_node} GPUs from amd-smi JSON, found {len(gpus)}",
                )

    def _evaluate_afm_devices(self, output: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        devices, errors = parse_afmctl_device_json(output)
        if len(devices) != self.expected_ifoe_devices_per_node:
            errors.append(f"Expected {self.expected_ifoe_devices_per_node} AFM devices, found {len(devices)}")
        accelerator_ids = set()
        vpod_sets = set()
        for device in devices:
            prefix = device.get("bdf", "<unknown BDF>")
            errors.extend(f"{prefix}: {error}" for error in device.get("parse_errors") or [])
            if device.get("config_phase") != "ACTIVE":
                errors.append(f"{prefix}: config_phase is {device.get('config_phase') or '<missing>'}, expected ACTIVE")
            if device.get("virtualization_mode") != self.expected_virtualization_mode:
                errors.append(
                    f"{prefix}: virtualization_mode is {device.get('virtualization_mode') or '<missing>'}, "
                    f"expected {self.expected_virtualization_mode}"
                )
            accelerator_id = device.get("accelerator_id")
            if accelerator_id is None:
                errors.append(f"{prefix}: missing accelerator_id")
            elif accelerator_id in accelerator_ids:
                errors.append(f"{prefix}: duplicate accelerator_id {accelerator_id}")
            else:
                accelerator_ids.add(accelerator_id)
            vpod = tuple(sorted(device.get("vpod_accelerators") or []))
            if not vpod:
                errors.append(f"{prefix}: missing vPOD accelerator membership")
            else:
                vpod_sets.add(vpod)
            port_count = device.get("num_network_ports")
            try:
                port_count = int(port_count)
            except (TypeError, ValueError):
                port_count = None
            if port_count != self.expected_network_ports_per_device:
                errors.append(
                    f"{prefix}: expected {self.expected_network_ports_per_device} network ports, found {port_count}"
                )
        if len(vpod_sets) > 1:
            errors.append("AFM devices disagree on vPOD accelerator membership")
        return devices, errors

    def _poll_afm_readiness(self) -> None:
        deadline = time.monotonic() + self.readiness_timeout_seconds
        attempt = 0
        latest_outputs: Dict[str, str] = {}
        latest_errors: Dict[str, List[str]] = {}
        while True:
            attempt += 1
            outputs = self._exec_all(self.build_afm_device_command())
            all_ready = True
            for node, result in self.results.items():
                raw = str(outputs.get(node, ""))
                latest_outputs[node] = raw
                devices, errors = self._evaluate_afm_devices(raw)
                result["afm_devices"] = devices
                result["raw_afm_devices"] = raw
                latest_errors[node] = errors
                if errors:
                    all_ready = False
            if all_ready or time.monotonic() >= deadline:
                break
            self.log_info(
                f"MI4XX AFM/vPOD admission pending after attempt {attempt}; retrying in {self.poll_interval_seconds}s"
            )
            time.sleep(self.poll_interval_seconds)
        for node, errors in latest_errors.items():
            result = self.results[node]
            result["afm_attempts"] = attempt
            for error in errors:
                self._add_error(result, error)

    def _evaluate_station_masks(self) -> None:
        outputs = self._exec_all(self.build_station_mask_command())
        all_masks: Dict[str, str] = {}
        for node, result in self.results.items():
            masks, errors = parse_station_masks(str(outputs.get(node, "")))
            result["station_masks"] = masks
            result["raw_station_masks"] = str(outputs.get(node, ""))
            for error in errors:
                self._add_error(result, error)
            if len(masks) != self.expected_gpus_per_node:
                self._add_error(result, f"Expected {self.expected_gpus_per_node} station masks, found {len(masks)}")
            for gpu_bdf, mask in masks.items():
                if self.reject_partial_stations and any(nibble in {"3", "c"} for nibble in mask):
                    self._add_error(result, f"{gpu_bdf}: partial IFoE station in mask {mask}")
                permitted = {"f", "0"} if self.allow_disabled_stations else {"f"}
                if not self.reject_partial_stations:
                    # This is retained for diagnostic profiles only.  A strict
                    # MI4XX benchmark profile should always reject c/3, because
                    # AFM does not expose a safe full-path interpretation for a
                    # half-enabled station.
                    permitted.update({"c", "3"})
                disallowed = sorted(set(mask) - permitted)
                if disallowed:
                    self._add_error(result, f"{gpu_bdf}: disallowed station-mask values {','.join(disallowed)}")
                enabled_ports = mask.count("f") * 2
                if enabled_ports < self.min_up_ports_per_gpu:
                    self._add_error(
                        result,
                        f"{gpu_bdf}: {enabled_ports} enabled IFoE ports is below min_up_ports_per_gpu="
                        f"{self.min_up_ports_per_gpu}",
                    )
                expected = self.expected_station_masks.get(gpu_bdf) or self.expected_station_masks.get(
                    gpu_bdf[:-1] + "1"
                )
                if expected and mask != expected:
                    self._add_error(result, f"{gpu_bdf}: station mask {mask} does not match configured {expected}")
                all_masks[f"{node}:{gpu_bdf}"] = mask
        if self.require_uniform_station_mask and all_masks:
            unique_masks = set(all_masks.values())
            if len(unique_masks) != 1:
                for result in self.results.values():
                    self._add_error(result, "Station masks are not uniform across selected MI4XX GPUs")

    def _cross_check_port_inventory(self) -> None:
        commands: Dict[str, str] = {}
        bdfs_by_node: Dict[str, List[str]] = {}
        for node, result in self.results.items():
            ifoe_bdfs = [
                gpu_bdf[:-1] + "1" for gpu_bdf in (result.get("station_masks") or {}) if gpu_bdf.endswith(".0")
            ]
            bdfs_by_node[node] = ifoe_bdfs
            commands[node] = self.build_afm_ports_command(ifoe_bdfs) if ifoe_bdfs else "true"
        outputs = self._exec_commands_by_host(commands)

        for node, result in self.results.items():
            inventories: Dict[str, Dict[str, Any]] = {}
            raw = outputs.get(node, "")
            parsed = AfmctlPortParser.parse(raw, allow_text_fallback=False)
            for error in parsed.get("parse_errors") or []:
                self._add_error(result, f"afmctl port JSON: {error}")

            ifoe_bdfs = bdfs_by_node.get(node, [])
            for gpu_bdf, mask in (result.get("station_masks") or {}).items():
                if not gpu_bdf.endswith(".0"):
                    continue
                ifoe_bdf = gpu_bdf[:-1] + "1"
                scoped = (parsed.get("ports_by_bdf") or {}).get(ifoe_bdf)
                if scoped is None and len(ifoe_bdfs) == 1:
                    scoped = parsed.get("unscoped_ports") or {}
                if scoped is None:
                    scoped = {}
                    self._add_error(result, f"{ifoe_bdf}: AFM port JSON contained no scoped inventory")

                expected_stations = {index for index, nibble in enumerate(mask) if nibble == "f"}
                ports_by_station: Dict[int, List[Dict[str, Any]]] = {}
                missing_station_ids = []
                for port in scoped.values():
                    station_id = port.get("station_id")
                    if station_id is None:
                        missing_station_ids.append(port.get("port"))
                        continue
                    ports_by_station.setdefault(station_id, []).append(port)
                if expected_stations and missing_station_ids:
                    self._add_error(
                        result,
                        f"{ifoe_bdf}: AFM port JSON omitted station_id for port(s) "
                        + ", ".join(str(port) for port in sorted(missing_station_ids)),
                    )

                enabled_station_ports = [
                    port for station in expected_stations for port in ports_by_station.get(station, [])
                ]
                up_enabled_station_ports = [port for port in enabled_station_ports if port.get("is_up")]
                inventories[ifoe_bdf] = {
                    "raw_output": raw,
                    "parsed": parsed,
                    "up_ports": len([port for port in scoped.values() if port.get("is_up")]),
                    "enabled_station_up_ports": len(up_enabled_station_ports),
                    "expected_enabled_ports": len(expected_stations) * 2,
                    "mask_enabled_port_ids": sorted(
                        port["port"] for port in enabled_station_ports if port.get("port") is not None
                    ),
                }
                for station in sorted(expected_stations):
                    station_ports = ports_by_station.get(station, [])
                    if len(station_ports) != 2:
                        self._add_error(
                            result,
                            f"{ifoe_bdf}: AFM reports {len(station_ports)} port(s) for enabled station "
                            f"{station}, expected 2",
                        )
                        continue
                    down_ports = [
                        f"{port.get('port')} ({port.get('state')})" for port in station_ports if not port.get("is_up")
                    ]
                    if down_ports:
                        self._add_error(
                            result,
                            f"{ifoe_bdf}: enabled station {station} has non-UP port(s): " + ", ".join(down_ports),
                        )
            result["afm_port_inventory"] = inventories

    def _reconcile_cluster_vpod(self) -> Dict[str, Any]:
        per_node: Dict[str, List[int]] = {}
        errors: List[str] = []
        for node, result in self.results.items():
            sets = {tuple(sorted(device.get("vpod_accelerators") or [])) for device in result.get("afm_devices") or []}
            sets.discard(())
            if len(sets) == 1:
                per_node[node] = list(next(iter(sets)))
            elif len(sets) > 1:
                errors.append(f"{node}: AFM devices report multiple vPOD memberships")
            else:
                errors.append(f"{node}: AFM reports no vPOD membership")
        unique_sets = {tuple(value) for value in per_node.values()}
        if len(unique_sets) > 1:
            errors.append("Selected nodes do not report the same AFM vPOD membership")
        return {
            "status": "PASS" if not errors and len(per_node) == len(self.results) else "FAIL",
            "per_node": per_node,
            "vpod_accelerators": list(next(iter(unique_sets))) if len(unique_sets) == 1 else [],
            "errors": errors,
            "source": "afmctl show device --json",
        }

    def run(self) -> Dict[str, Any]:
        hosts = list(getattr(self.phdl, "reachable_hosts", []) or [])
        self.results = {
            node: {
                "status": "PASS",
                "errors": [],
                "gpu_inventory": [],
                "afm_devices": [],
                "station_masks": {},
                "afm_port_inventory": {},
                "kernel_failures": [],
            }
            for node in hosts
        }
        if not hosts:
            return {
                "status": "FAIL",
                "fabric_checks": self.fabric_checks,
                "node_results": {},
                "vpod_membership": {
                    "status": "FAIL" if self.fabric_checks else "SKIPPED",
                    "errors": ["No reachable nodes"] if self.fabric_checks else [],
                },
                "errors": ["No reachable nodes for node-health admission"],
            }

        host_outputs: Dict[str, Dict[str, str]] = {node: {} for node in hosts}
        host_outputs_map = {
            "amdgpu": self._exec_all(self._module_command("amdgpu")),
            "kfd": self._exec_all("test -e /dev/kfd && printf 1 || printf 0"),
            "kernel": self._exec_all(self._kernel_command()),
        }
        if self.fabric_checks:
            host_outputs_map["agent"] = self._exec_all(self._agent_command(self.agent_process_name))
            for module in self.required_ifoe_modules:
                host_outputs_map[f"module:{module}"] = self._exec_all(self._module_command(module))
        for key, outputs in host_outputs_map.items():
            for node in hosts:
                host_outputs[node][key] = str(outputs.get(node, ""))
        self._evaluate_host_prerequisites(host_outputs)
        self._evaluate_gpu_inventory(self._exec_all(self.build_gpu_inventory_command()))
        if self.fabric_checks:
            self._poll_afm_readiness()
            self._evaluate_station_masks()
            self._cross_check_port_inventory()
            membership = self._reconcile_cluster_vpod()
            for error in membership["errors"]:
                for result in self.results.values():
                    self._add_error(result, error)
        else:
            membership = {
                "status": "SKIPPED",
                "skipped": True,
                "vpod_accelerators": [],
                "per_node": {},
                "errors": [],
                "message": "MI4XX fabric checks are disabled",
            }
        failures = [node for node, result in self.results.items() if result["status"] == "FAIL"]
        return {
            "status": "FAIL" if failures or (self.fabric_checks and membership["status"] != "PASS") else "PASS",
            "fabric_checks": self.fabric_checks,
            "node_results": self.results,
            "failed_nodes": failures,
            "vpod_membership": membership,
            "errors": membership["errors"],
        }


# Backward-compatible library alias; customer configuration now uses the
# generation-neutral ``node_health`` name.
Mi4xxNodeHealthCheck = NodeHealthCheck


__all__ = [
    "NodeHealthCheck",
    "Mi4xxNodeHealthCheck",
    "parse_afmctl_device_json",
    "parse_amd_smi_gpu_json",
    "parse_station_masks",
]
