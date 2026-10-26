'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Per-node hardware topology probe.

Builds one shell command that emits delimited sections, run once across every
node with a single orch.exec call, then parses each host's output into a plain
node-topology dict. The probe anchors on amd-smi structured JSON for the GPU
XGMI mesh and on sysfs for CPU/NUMA/NIC. Nothing here requires a live cluster:
build_probe_cmd is a pure string builder and parse_node_topology works on
captured text, so both are unit-tested against recorded node output.

Design notes grounded in real node output (verified Oct 2026):
  - `amd-smi list --json` emits SEVERAL concatenated top-level JSON arrays (the
    GPU array, then an ai_nic array), so a plain json.loads over the whole blob
    raises "Extra data". _load_json_documents walks every top-level document.
  - On a node whose amdgpu driver is down (no /dev/kfd), amd-smi prints nothing
    on stdout; the node still has CPU/NUMA/NIC from sysfs. That node parses to a
    record with gpus=[] rather than raising -- the key graceful-degradation path.
'''

import json
import re

# Section markers. Chosen to be shell-safe and never appear in tool output.
SEC_HOSTNAME = "@@CVS_TOPO_HOSTNAME@@"
SEC_AMDSMI_TOPOLOGY = "@@CVS_TOPO_AMDSMI_TOPOLOGY@@"
SEC_AMDSMI_LIST = "@@CVS_TOPO_AMDSMI_LIST@@"
SEC_LSCPU = "@@CVS_TOPO_LSCPU@@"
SEC_NUMA = "@@CVS_TOPO_NUMA@@"
SEC_NICS = "@@CVS_TOPO_NICS@@"
SEC_DIMMS = "@@CVS_TOPO_DIMMS@@"
SEC_NVME = "@@CVS_TOPO_NVME@@"
SEC_PARTITION = "@@CVS_TOPO_PARTITION@@"
SEC_RDMA = "@@CVS_TOPO_RDMA@@"
SEC_LLDP = "@@CVS_TOPO_LLDP@@"
SEC_END = "@@CVS_TOPO_END@@"

DEFAULT_PROBE_TIMEOUT = 120


def build_probe_cmd():
    '''Return the shell command that prints every topology section for one node.

    Every section is bracketed by its marker and the next marker, so the parser
    can slice the output even when an individual tool prints nothing (missing
    amd-smi, driver down). The command never fails the whole node on one missing
    tool: each probe is guarded with `|| true` / `2>/dev/null`.
    '''
    lines = [
        f"echo {SEC_HOSTNAME}",
        "hostname",
        f"echo {SEC_AMDSMI_TOPOLOGY}",
        "amd-smi topology --json 2>/dev/null || true",
        f"echo {SEC_AMDSMI_LIST}",
        "amd-smi list --json 2>/dev/null || true",
        f"echo {SEC_LSCPU}",
        "lscpu 2>/dev/null || true",
        f"echo {SEC_NUMA}",
        # One "node<N>: <cpulist>" line per populated NUMA node.
        (
            'for n in /sys/devices/system/node/node[0-9]*; do '
            'cl=$(cat "$n/cpulist" 2>/dev/null); '
            '[ -n "$cl" ] && echo "$(basename "$n"): $cl"; done'
        ),
        f"echo {SEC_NICS}",
        # One line per non-loopback interface: name, pci bdf, numa, operstate.
        (
            'for d in /sys/class/net/*; do nm=$(basename "$d"); '
            '[ "$nm" = "lo" ] && continue; '
            'bdf=$(basename "$(readlink "$d/device" 2>/dev/null)" 2>/dev/null); '
            'echo "$nm bdf=$bdf numa=$(cat "$d/device/numa_node" 2>/dev/null) '
            'oper=$(cat "$d/operstate" 2>/dev/null)"; done'
        ),
        f"echo {SEC_DIMMS}",
        # Per-DIMM detail needs SMBIOS, which is root-only. Emit the raw
        # `dmidecode -t 17` blocks when passwordless sudo is available, else fall
        # back to lsmem summary lines prefixed LSMEM: so the parser can tell them
        # apart. `-q` suppresses dmidecode's version banner and the harmless
        # "SMBIOS newer than 3.7.0 not fully supported" note. Never prompts
        # (sudo -n) and never fails the probe.
        (
            'if sudo -n true 2>/dev/null; then sudo -n dmidecode -q -t 17 2>/dev/null; '
            'else lsmem 2>/dev/null | sed "s/^/LSMEM: /"; fi || true'
        ),
        f"echo {SEC_NVME}",
        # One line per NVMe namespace from sysfs (no sudo): name, 512B-sector
        # count, model, controller NUMA node.
        (
            'for d in /sys/block/nvme*; do [ -e "$d" ] || continue; '
            'nm=$(basename "$d"); '
            'echo "$nm sectors=$(cat "$d/size" 2>/dev/null) '
            'numa=$(cat "$d/device/device/numa_node" 2>/dev/null) '
            'model=$(cat "$d/device/model" 2>/dev/null)"; done'
        ),
        f"echo {SEC_PARTITION}",
        # GPU partition mode (SPX/CPX + NPSx) governs scale-up sharding.
        "amd-smi partition --json 2>/dev/null || true",
        f"echo {SEC_RDMA}",
        # One line per RDMA device: the scale-out fabric descriptor. link_layer
        # Ethernet=RoCE / InfiniBand=IB; rate is the per-NIC bandwidth; bdf ties
        # it back to a GPU's PCI domain. No sudo.
        (
            'for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue; '
            'nm=$(basename "$d"); '
            'bdf=$(basename "$(readlink "$d/device" 2>/dev/null)" 2>/dev/null); '
            'netdev=$(ls "$d/device/net" 2>/dev/null | head -1); '
            'echo "rdma=$nm netdev=$netdev bdf=$bdf '
            'link_layer=$(cat "$d/ports/1/link_layer" 2>/dev/null) '
            'state=$(cat "$d/ports/1/state" 2>/dev/null) '
            'rate=[$(cat "$d/ports/1/rate" 2>/dev/null)]"; done'
        ),
        f"echo {SEC_LLDP}",
        # Inter-node wiring: LLDP neighbor (switch name + port) per interface.
        # Needs an lldpd daemon; absent on many nodes, so this is best-effort and
        # the parser treats an empty result as "switch topology not discovered".
        # Tries unprivileged first, then sudo -n; never prompts, never fails.
        ('lldpctl -f keyvalue 2>/dev/null || sudo -n lldpctl -f keyvalue 2>/dev/null || true'),
        f"echo {SEC_END}",
    ]
    return "; ".join(lines)


def _slice_sections(raw):
    '''Split one node's raw probe output into {marker: section_text}.

    Tolerant of leading banner noise (SSH MOTD) before the first marker and of a
    missing trailing SEC_END. A marker that never appears yields "".
    '''
    order = [
        SEC_HOSTNAME,
        SEC_AMDSMI_TOPOLOGY,
        SEC_AMDSMI_LIST,
        SEC_LSCPU,
        SEC_NUMA,
        SEC_NICS,
        SEC_DIMMS,
        SEC_NVME,
        SEC_PARTITION,
        SEC_RDMA,
        SEC_LLDP,
        SEC_END,
    ]
    sections = {m: "" for m in order}
    positions = []
    for marker in order:
        idx = raw.find(marker)
        if idx != -1:
            positions.append((idx, marker))
    positions.sort()
    for i, (idx, marker) in enumerate(positions):
        start = idx + len(marker)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(raw)
        sections[marker] = raw[start:end].strip("\n")
    return sections


def _load_json_documents(text):
    '''Return a list of every top-level JSON value concatenated in text.

    amd-smi concatenates multiple JSON arrays on stdout; json.JSONDecoder.raw_decode
    lets us consume them one at a time. Non-JSON leading/trailing noise is skipped.
    Returns [] when nothing parses.
    '''
    decoder = json.JSONDecoder()
    docs = []
    pos = 0
    length = len(text)
    while pos < length:
        # Skip to the next plausible JSON start.
        while pos < length and text[pos] not in "[{":
            pos += 1
        if pos >= length:
            break
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        docs.append(value)
        pos = end
    return docs


def _parse_amdsmi_list(section_text):
    '''From `amd-smi list --json`, return (gpus, ai_nics).

    A GPU record has a "gpu" key; an ai_nic record has "ai_nic". Multiple arrays
    are flattened and classified by key so ordering changes do not matter.
    '''
    gpus = []
    ai_nics = []
    for doc in _load_json_documents(section_text):
        items = doc if isinstance(doc, list) else [doc]
        for item in items:
            if not isinstance(item, dict):
                continue
            if "gpu" in item:
                gpus.append(item)
            elif "ai_nic" in item:
                ai_nics.append(item)
    return gpus, ai_nics


def _parse_amdsmi_topology(section_text):
    '''From `amd-smi topology --json`, return the per-GPU link records list.

    Each entry is {"gpu", "bdf", "links": [...]} where every link carries
    link_type (SELF/XGMI/PCIE), weight, num_hops, link_status. Returns [] when
    the section is empty (driver down).
    '''
    for doc in _load_json_documents(section_text):
        if isinstance(doc, list):
            return doc
    return []


_LSCPU_KEYS = {
    "CPU(s)": "cpus",
    "Model name": "model_name",
    "Socket(s)": "sockets",
    "Core(s) per socket": "cores_per_socket",
    "Thread(s) per core": "threads_per_core",
    "NUMA node(s)": "numa_nodes",
    "Vendor ID": "vendor",
}


def _parse_lscpu(section_text):
    '''Parse the handful of lscpu fields we surface into a flat dict.'''
    cpu = {}
    for line in section_text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in _LSCPU_KEYS:
            cpu[_LSCPU_KEYS[key]] = value.strip()
    return cpu


_NUMA_RE = re.compile(r"^(node\d+):\s*(.+)$")


def _parse_numa(section_text):
    '''Parse "node<N>: <cpulist>" lines into [{"node", "cpulist"}].'''
    numa = []
    for line in section_text.splitlines():
        match = _NUMA_RE.match(line.strip())
        if match:
            numa.append({"node": match.group(1), "cpulist": match.group(2).strip()})
    return numa


_NIC_RE = re.compile(r"^(?P<name>\S+)\s+bdf=(?P<bdf>\S*)\s+numa=(?P<numa>\S*)\s+oper=(?P<oper>\S*)$")


def _parse_nics(section_text):
    '''Parse the per-NIC probe lines into [{"name","bdf","numa_node","operstate"}].'''
    nics = []
    for line in section_text.splitlines():
        match = _NIC_RE.match(line.strip())
        if not match:
            continue
        nics.append(
            {
                "name": match.group("name"),
                "bdf": match.group("bdf") or None,
                "numa_node": match.group("numa") or None,
                "operstate": match.group("oper") or None,
            }
        )
    return nics


def _parse_dmidecode_dimms(section_text):
    '''Parse `dmidecode -t 17` output into populated system-DIMM records.

    Each "Memory Device" block is one slot. Skipped: slots with "Size: No Module
    Installed" (or no size), and on-package GPU HBM, which SMBIOS also reports as
    a memory device (Form Factor "Die", Locator "HBM") -- that is GPU memory,
    already represented by the GPU nodes, not a system DIMM. Returns
    [{"locator","bank","size","type","speed","manufacturer","part_number","rank"}].
    '''
    dimms = []
    block = {}

    def flush():
        size = block.get("Size", "")
        if not size or "No Module" in size or size.lower() == "unknown":
            return
        form = (block.get("Form Factor") or "").strip().lower()
        if form == "die" or (block.get("Locator") or "").strip().upper() == "HBM":
            return  # GPU HBM, not a system DIMM
        dimms.append(
            {
                "locator": (block.get("Locator") or "").strip() or None,
                "bank": (block.get("Bank Locator") or "").strip() or None,
                "size": size.strip(),
                "type": (block.get("Type") or "").strip() or None,
                "speed": (block.get("Speed") or "").strip() or None,
                "manufacturer": (block.get("Manufacturer") or "").strip() or None,
                "part_number": (block.get("Part Number") or "").strip() or None,
                "rank": (block.get("Rank") or "").strip() or None,
            }
        )

    for line in section_text.splitlines():
        if line.startswith("Memory Device"):
            flush()
            block = {}
            continue
        if ":" in line and (line.startswith("\t") or line.startswith(" ")):
            key, _, value = line.strip().partition(":")
            block[key.strip()] = value.strip()
    flush()
    return dimms


def _parse_lsmem_summary(section_text):
    '''Fallback memory summary from lsmem (LSMEM:-prefixed lines).

    Used when sudo/dmidecode is unavailable, so per-DIMM detail is impossible.
    Returns {"total_online": <str or None>} or {} when no lsmem lines present.
    '''
    total = None
    for line in section_text.splitlines():
        if not line.startswith("LSMEM:"):
            continue
        body = line[len("LSMEM:") :].strip()
        if body.lower().startswith("total online memory"):
            total = body.split(":", 1)[1].strip() if ":" in body else None
    return {"total_online": total} if total else {}


def _parse_dimms(section_text):
    '''Return (dimms, mem_summary) from the DIMM section.

    dimms is the populated-DIMM list from dmidecode (possibly empty); mem_summary
    is the lsmem fallback dict used when dmidecode produced nothing (no sudo).
    '''
    dimms = _parse_dmidecode_dimms(section_text)
    if dimms:
        return dimms, {}
    return [], _parse_lsmem_summary(section_text)


_NVME_RE = re.compile(r"^(?P<name>\S+)\s+sectors=(?P<sectors>\S*)\s+numa=(?P<numa>\S*)\s+model=(?P<model>.*)$")


def _nvme_capacity(sectors):
    '''Human capacity from a 512-byte sector count, e.g. "3.8 TB". None if unknown.'''
    try:
        total = int(sectors) * 512
    except (TypeError, ValueError):
        return None
    tb = total / 1_000_000_000_000
    if tb >= 1:
        return f"{tb:.1f} TB"
    gb = total / 1_000_000_000
    return f"{gb:.0f} GB"


def _parse_nvme(section_text):
    '''Parse NVMe probe lines into [{"name","capacity","numa_node","model"}].'''
    nvme = []
    for line in section_text.splitlines():
        match = _NVME_RE.match(line.strip())
        if not match:
            continue
        nvme.append(
            {
                "name": match.group("name"),
                "capacity": _nvme_capacity(match.group("sectors")),
                "numa_node": match.group("numa") or None,
                "model": match.group("model").strip() or None,
            }
        )
    return nvme


def _parse_partition(section_text):
    '''From `amd-smi partition --json`, return (per_gpu, node_mode).

    per_gpu is [{"gpu_id","accelerator_type","memory"}] (e.g. SPX, NPS1);
    node_mode is "<accelerator_type>/<memory>" when every GPU agrees (the common
    case), "mixed" when they differ, or None when nothing parsed. The partition
    mode governs scale-up sharding, so it is surfaced at node level.
    '''
    per_gpu = []
    for doc in _load_json_documents(section_text):
        if not isinstance(doc, dict):
            continue
        for entry in doc.get("current_partition", []):
            if not isinstance(entry, dict):
                continue
            per_gpu.append(
                {
                    "gpu_id": entry.get("gpu_id"),
                    "accelerator_type": entry.get("accelerator_type"),
                    "memory": entry.get("memory"),
                }
            )
    if not per_gpu:
        return [], None
    modes = {f'{g["accelerator_type"]}/{g["memory"]}' for g in per_gpu}
    node_mode = modes.pop() if len(modes) == 1 else "mixed"
    return per_gpu, node_mode


_RDMA_RE = re.compile(
    r"^rdma=(?P<rdma>\S+)\s+netdev=(?P<netdev>\S*)\s+bdf=(?P<bdf>\S*)\s+"
    r"link_layer=(?P<ll>\S*)\s+state=(?P<state>.*?)\s+rate=\[(?P<rate>.*)\]$"
)

# amd-smi/ib rate strings like "400 Gb/sec (4X NDR)". Pull the leading number+unit
# and the generation tag for a compact scale-out label.
_RATE_RE = re.compile(r"(?P<gbps>\d+)\s*Gb/sec(?:\s*\((?:\d+X\s*)?(?P<gen>[A-Z]+)\))?")


def _fabric_from_link_layer(link_layer):
    '''RoCE for Ethernet RDMA, IB for InfiniBand, else the raw value.'''
    ll = (link_layer or "").strip().lower()
    if ll == "ethernet":
        return "RoCE"
    if ll == "infiniband":
        return "IB"
    return link_layer or None


def _parse_rdma(section_text):
    '''Parse RDMA probe lines into scale-out NIC records.

    Returns [{"rdma","netdev","bdf","link_layer","fabric","state","active",
    "rate","gbps"}]. fabric is RoCE/IB; gbps is the integer line rate; active is
    True when the port state contains ACTIVE. bdf ties each NIC to a GPU domain.
    '''
    nics = []
    for line in section_text.splitlines():
        match = _RDMA_RE.match(line.strip())
        if not match:
            continue
        rate = match.group("rate").strip()
        rate_match = _RATE_RE.search(rate)
        state = match.group("state").strip()
        nics.append(
            {
                "rdma": match.group("rdma"),
                "netdev": match.group("netdev") or None,
                "bdf": match.group("bdf") or None,
                "link_layer": match.group("ll") or None,
                "fabric": _fabric_from_link_layer(match.group("ll")),
                "state": state or None,
                "active": "ACTIVE" in state.upper(),
                "rate": rate or None,
                "gbps": int(rate_match.group("gbps")) if rate_match else None,
                "gen": rate_match.group("gen") if rate_match else None,
            }
        )
    return nics


def _parse_lldp(section_text):
    '''Parse `lldpctl -f keyvalue` into {ifname: {switch_name, switch_port}}.

    keyvalue lines look like `lldp.<ifname>.chassis.name=<switch>` and
    `lldp.<ifname>.port.descr=<port>`. Returns {} when no neighbors are present
    (no lldpd daemon) -- the common case, treated as "not discovered" upstream.
    '''
    neighbors = {}
    for line in section_text.splitlines():
        line = line.strip()
        if not line.startswith("lldp.") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parts = key.split(".")
        if len(parts) < 4:
            continue
        ifname = parts[1]
        rest = ".".join(parts[2:])
        entry = neighbors.setdefault(ifname, {})
        if rest in ("chassis.name", "chassis.id"):
            entry.setdefault("switch_name", value.strip())
        elif rest in ("port.descr", "port.id"):
            entry.setdefault("switch_port", value.strip())
    # Keep only entries that named a switch.
    return {k: v for k, v in neighbors.items() if v.get("switch_name")}


def _build_gpu_records(list_gpus, topology_gpus):
    '''Merge amd-smi list identity with amd-smi topology links into GPU records.

    Keyed by bdf. A GPU present in either source appears; links come from the
    topology source, identity (uuid/kfd_id/node_id) from the list source.
    '''
    by_bdf = {}
    for g in list_gpus:
        bdf = g.get("bdf")
        if not bdf:
            continue
        by_bdf[bdf] = {
            "bdf": bdf,
            "gpu_index": g.get("gpu"),
            "uuid": g.get("uuid"),
            "kfd_id": g.get("kfd_id"),
            "node_id": g.get("node_id"),
            "partition_id": g.get("partition_id"),
            "links": [],
        }
    for t in topology_gpus:
        bdf = t.get("bdf")
        if not bdf:
            continue
        rec = by_bdf.setdefault(
            bdf,
            {
                "bdf": bdf,
                "gpu_index": t.get("gpu"),
                "uuid": None,
                "kfd_id": None,
                "node_id": None,
                "partition_id": None,
                "links": [],
            },
        )
        rec["links"] = [
            {
                "peer_bdf": link.get("bdf"),
                "peer_gpu": link.get("gpu"),
                "link_type": link.get("link_type"),
                "weight": link.get("weight"),
                "num_hops": link.get("num_hops"),
                "link_status": link.get("link_status"),
            }
            for link in t.get("links", [])
        ]
    # Stable order by GPU index then bdf.
    return sorted(
        by_bdf.values(),
        key=lambda r: (r["gpu_index"] if r["gpu_index"] is not None else 1 << 30, r["bdf"]),
    )


def parse_node_topology(raw_output, host=None):
    '''Parse one node's raw probe output into a plain topology dict.

    Returns a record shaped:
      {
        "host": <str or None>,        # the cluster_file key, if provided
        "hostname": <str or None>,    # as reported by `hostname`
        "reachable": <bool>,          # False when the output has no usable sections
        "cpu": {...}, "numa": [...], "nics": [...], "gpus": [...],
        "dimms": [...],               # populated system DIMMs (dmidecode, sudo)
        "mem_summary": {...},         # lsmem fallback when dmidecode unavailable
        "nvme": [...],                # NVMe namespaces (sysfs)
        "partition": [...],           # per-GPU partition (amd-smi)
        "scale_up_mode": <str>,       # e.g. "SPX/NPS1" -- scale-up sharding mode
        "rdma": [...],                # scale-out NIC fabric (RDMA sysfs)
        "lldp": {...},                # ifname -> {switch_name, switch_port} (best-effort)
        "gpu_count": <int>,
      }

    Never raises on malformed/partial input: a node whose probe returned nothing
    useful comes back reachable=False; a node with CPU/NIC data but no GPUs
    (driver down) comes back reachable=True with gpus=[].
    '''
    raw = raw_output or ""
    sections = _slice_sections(raw)

    hostname = sections.get(SEC_HOSTNAME, "").strip() or None
    cpu = _parse_lscpu(sections.get(SEC_LSCPU, ""))
    numa = _parse_numa(sections.get(SEC_NUMA, ""))
    nics = _parse_nics(sections.get(SEC_NICS, ""))
    dimms, mem_summary = _parse_dimms(sections.get(SEC_DIMMS, ""))
    nvme = _parse_nvme(sections.get(SEC_NVME, ""))
    partition, scale_up_mode = _parse_partition(sections.get(SEC_PARTITION, ""))
    rdma = _parse_rdma(sections.get(SEC_RDMA, ""))
    lldp = _parse_lldp(sections.get(SEC_LLDP, ""))
    list_gpus, ai_nics = _parse_amdsmi_list(sections.get(SEC_AMDSMI_LIST, ""))
    topology_gpus = _parse_amdsmi_topology(sections.get(SEC_AMDSMI_TOPOLOGY, ""))
    gpus = _build_gpu_records(list_gpus, topology_gpus)

    # Reachable means we got at least one real section back. A node that returned
    # only banner noise (SSH failure captured as text) has no markers at all.
    reachable = bool(hostname or cpu or numa or nics or gpus or dimms or nvme or rdma)

    return {
        "host": host,
        "hostname": hostname,
        "reachable": reachable,
        "cpu": cpu,
        "numa": numa,
        "nics": nics,
        "ai_nics": ai_nics,
        "dimms": dimms,
        "mem_summary": mem_summary,
        "nvme": nvme,
        "partition": partition,
        "scale_up_mode": scale_up_mode,
        "rdma": rdma,
        "lldp": lldp,
        "gpus": gpus,
        "gpu_count": len(gpus),
    }
