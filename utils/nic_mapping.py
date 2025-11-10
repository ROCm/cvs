#!/usr/bin/env python3
"""
Generic NIC ? GPU/Accelerator Mapping (Linux, /sys)
- NICs: PCI class 0x02*
- GPUs: PCI class 0x03*
- (optional) Accelerators: PCI class 0x12* with --include-accelerators
Scoring priority (lower is better):
  1) NUMA mismatch penalty (0 if match, 1 otherwise)
  2) LCA hops (max hops up to LCA from either side)
  3) Total hops (up+up)
  4) Tie-breaker: device driver presence (prefer bound driver)

Usage:
  python3 nic_gpu_map_generic.py [--json | --csv FILE] [--include-accelerators]
                                 [--nic-vendor 0x14e4 --nic-vendor 0x8086 ...]
                                 [--gpu-vendor 0x10de --gpu-vendor 0x1002 ...]
                                 [--prefer-same-switch] [--one-to-one] [--max-gpu-per-nic N]

Examples:
  # Simple table (all vendors)
  python3 nic_gpu_map_generic.py

  # JSON output with accelerators included
  python3 nic_gpu_map_generic.py --include-accelerators --json

  # CSV restricted to Broadcom/Intel NICs and NVIDIA/AMD GPUs
  python3 nic_gpu_map_generic.py --csv map.csv \
      --nic-vendor 0x14e4 --nic-vendor 0x8086 \
      --gpu-vendor 0x10de --gpu-vendor 0x1002

  # Map up to 2 GPUs per NIC, ensuring same NUMA and closest group
  python3 nic_gpu_map_generic.py --max-gpu-per-nic 2
"""

import os
import re
import csv
import json
import argparse
import subprocess
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

SYS_PCI = "/sys/bus/pci/devices"
BDF_RE = re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")

def read_text(path: str, default: Optional[str] = None) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return default

def list_bdfs() -> List[str]:
    if not os.path.isdir(SYS_PCI):
        return []
    return [d for d in os.listdir(SYS_PCI) if BDF_RE.match(d)]

def lspci_name(addr: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["lspci", "-s", addr], text=True).strip()
        return re.sub(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\s*", "", out)
    except Exception:
        return None

def device_info(addr: str, known_bdfs: set) -> Dict:
    base = os.path.join(SYS_PCI, addr)
    info = {
        "addr": addr,
        "vendor": read_text(os.path.join(base, "vendor")),   # e.g. 0x14e4
        "device": read_text(os.path.join(base, "device")),   # e.g. 0x1750
        "class":  read_text(os.path.join(base, "class")),    # e.g. 0x020000
        "numa":   read_text(os.path.join(base, "numa_node")),
        "driver": None,
        "name": None,
        "parent": None,
        "is_vf": False,
        "pf_addr": None,
        "linux_name": None,
        "ib_name": None,
    }
    drv_link = os.path.join(base, "driver")
    if os.path.islink(drv_link):
        info["driver"] = os.path.basename(os.path.realpath(drv_link))

    info["name"] = lspci_name(addr)

    # Linux interface name (for NICs)
    net_dir = os.path.join(base, "net")
    if os.path.isdir(net_dir):
        try:
            interfaces = os.listdir(net_dir)
            if interfaces:
                info["linux_name"] = interfaces[0]
        except Exception:
            pass

    # InfiniBand device name
    ib_dir = os.path.join(base, "infiniband")
    if os.path.isdir(ib_dir):
        try:
            ib_devices = os.listdir(ib_dir)
            if ib_devices:
                info["ib_name"] = ib_devices[0]
        except Exception:
            pass

    # Parent device (walk one level up the resolved path)
    pdir = os.path.dirname(os.path.realpath(base))
    parent = os.path.basename(pdir)
    info["parent"] = parent if parent in known_bdfs else None

    # SR-IOV PF link if VF
    physfn = os.path.join(base, "physfn")
    if os.path.islink(physfn):
        info["is_vf"] = True
        info["pf_addr"] = os.path.basename(os.path.realpath(physfn))

    # Normalize NUMA
    try:
        info["numa"] = int(info["numa"])
    except Exception:
        info["numa"] = -1

    return info

def load_devices() -> Dict[str, Dict]:
    bdfs = list_bdfs()
    known = set(bdfs)
    devs = {bdf: device_info(bdf, known) for bdf in bdfs}
    # Re-check parents after initial load (in case ordering matters)
    for bdf, d in devs.items():
        base = os.path.join(SYS_PCI, bdf)
        pdir = os.path.dirname(os.path.realpath(base))
        parent = os.path.basename(pdir)
        d["parent"] = parent if parent in devs else None
    return devs

def is_class(d: Dict, prefix: str) -> bool:
    c = d.get("class") or ""
    return c.startswith(prefix)

def is_nic(d: Dict) -> bool:
    return is_class(d, "0x02")  # network controller

def is_gpu_or_display(d: Dict) -> bool:
    return is_class(d, "0x03")  # display/VGA/3D controller

def is_accelerator(d: Dict) -> bool:
    return is_class(d, "0x12")  # processing accelerators (optional)

def filter_by_vendor(devs: List[Dict], allowed: Optional[List[str]]) -> List[Dict]:
    if not allowed:
        return devs
    allowed_l = {v.lower() for v in allowed}
    return [d for d in devs if (d.get("vendor") or "").lower() in allowed_l]

def ancestry_chain(devs: Dict[str, Dict], addr: str) -> List[str]:
    chain = []
    cur = addr
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = devs.get(cur, {}).get("parent")
    return chain

def lowest_common_ancestor(devs: Dict[str, Dict], a: str, b: str) -> Optional[str]:
    ac = ancestry_chain(devs, a)
    bc = ancestry_chain(devs, b)
    aset = set(ac)
    for node in bc:
        if node in aset:
            return node
    return None

def hop_metrics(devs: Dict[str, Dict], a: str, b: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    ac = ancestry_chain(devs, a)
    bc = ancestry_chain(devs, b)
    lca = lowest_common_ancestor(devs, a, b)
    if lca is None:
        return None, None, None, None
    da = ac.index(lca)
    db = bc.index(lca)
    return da + db, da, db, lca

def score_pair(nic: Dict, gpu: Dict, devs: Dict[str, Dict], prefer_same_switch: bool = False) -> Tuple[int, int, int, int]:
    """
    Lower is better.
    Returns tuple:
      (numa_penalty, lca_hops, total_hops, driver_tiebreak)
    If prefer_same_switch: amplify LCA difference to favor closer switch-level proximity.
    """
    topo = hop_metrics(devs, nic["addr"], gpu["addr"])
    if topo[0] is None:
        return (999999, 999999, 999999, 1)
    total, da, db, _ = topo
    lca_hops = max(da, db)
    if prefer_same_switch:
        # Heavier weight on lca_hops to push same-port/same-switch pairings
        lca_hops *= 2

    numa_penalty = 0 if (nic["numa"] != -1 and nic["numa"] == gpu["numa"]) else 1
    driver_tiebreak = 0 if gpu.get("driver") else 1
    return (numa_penalty, lca_hops, total, driver_tiebreak)

def pick_mappings(devs: Dict[str, Dict],
                  nic_vendors: Optional[List[str]],
                  gpu_vendors: Optional[List[str]],
                  include_accels: bool,
                  prefer_same_switch: bool,
                  one_to_one: bool,
                  max_gpus_per_nic: int = 1) -> List[Dict]:
    nics = [d for d in devs.values() if is_nic(d)]
    gpus = [d for d in devs.values() if (is_gpu_or_display(d) or (include_accels and is_accelerator(d)))]

    nics = filter_by_vendor(nics, nic_vendors)
    gpus = filter_by_vendor(gpus, gpu_vendors)

    # Group by NUMA node
    numa_groups = {}
    for d in nics + gpus:
        numa = d.get("numa", -1)
        if numa not in numa_groups:
            numa_groups[numa] = {"nics": [], "gpus": []}
        if is_nic(d):
            numa_groups[numa]["nics"].append(d)
        else:
            numa_groups[numa]["gpus"].append(d)

    assignments = []
    used_gpus = set()

    for numa, group in numa_groups.items():

        group_nics = group["nics"]
        group_gpus = group["gpus"]

        # SR-IOV handling
        topo_alias = {n["addr"]: n["pf_addr"] if n["is_vf"] and n["pf_addr"] else n["addr"] for n in group_nics}

        # Sort NICs by potential (e.g., number of available GPUs) to prioritize
        group_nics.sort(key=lambda n: len([g for g in group_gpus if g["addr"] not in used_gpus]), reverse=True)

        for n in group_nics:
            nic_topo_addr = topo_alias[n["addr"]]
            nic_topo = dict(n)
            nic_topo["addr"] = nic_topo_addr

            available_gpus = [g for g in group_gpus if g["addr"] not in used_gpus]

            if max_gpus_per_nic == 1:
                # Original logic
                best = None
                best_score = None
                for g in available_gpus:
                    if one_to_one and g["addr"] in used_gpus:
                        continue
                    s = score_pair(nic_topo, g, devs, prefer_same_switch)
                    if (best_score is None) or (s < best_score):
                        best_score = s
                        best = g
                if best:
                    total, da, db, lca = hop_metrics(devs, nic_topo_addr, best["addr"])
                    assignments.append({
                        "nic_addr": n["addr"],
                        "nic_name": n.get("name") or n["addr"],
                        "nic_driver": n.get("driver"),
                        "nic_vendor": n.get("vendor"),
                        "nic_device": n.get("device"),
                        "nic_numa": n.get("numa"),
                        "nic_linux_name": n.get("linux_name"),
                        "nic_ib_name": n.get("ib_name"),
                        "gpu_addr": best["addr"],
                        "gpu_name": best.get("name") or best["addr"],
                        "gpu_driver": best.get("driver"),
                        "gpu_vendor": best.get("vendor"),
                        "gpu_device": best.get("device"),
                        "gpu_numa": best.get("numa"),
                        "lca": lca,
                        "hops_total": total,
                        "nic_up_hops": da,
                        "gpu_up_hops": db,
                        "numa_match": True,  # All in same NUMA
                        "score": best_score,
                        "topology_nic_node": nic_topo_addr if nic_topo_addr != n["addr"] else None,
                        "is_vf": n.get("is_vf", False),
                        "pf_addr": n.get("pf_addr"),
                        "group_max_hops": None,  # Not applicable for single
                    })
                    if one_to_one:
                        used_gpus.add(best["addr"])
            elif max_gpus_per_nic >= 2:
                # Group GPUs by their LCA with the NIC
                from collections import defaultdict
                lca_groups = defaultdict(list)
                for g in available_gpus:
                    lca = lowest_common_ancestor(devs, nic_topo_addr, g["addr"])
                    if lca:
                        lca_groups[lca].append(g)

                # Find the best LCA group with enough GPUs
                best_group = None
                best_max_hops = float('inf')
                for lca, gpus in lca_groups.items():
                    if len(gpus) >= max_gpus_per_nic:
                        # Sort by total hops to NIC
                        gpus.sort(key=lambda g: hop_metrics(devs, nic_topo_addr, g["addr"])[0] or 999999)
                        selected = gpus[:max_gpus_per_nic]
                        # Compute max hops in group
                        max_hops = 0
                        for g in selected:
                            hops = hop_metrics(devs, nic_topo_addr, g["addr"])[0]
                            if hops:
                                max_hops = max(max_hops, hops)
                        for i in range(len(selected)):
                            for j in range(i+1, len(selected)):
                                hops = hop_metrics(devs, selected[i]["addr"], selected[j]["addr"])[0]
                                if hops:
                                    max_hops = max(max_hops, hops)
                        if max_hops < best_max_hops:
                            best_max_hops = max_hops
                            best_group = (lca, selected)

                if best_group:
                    lca, selected = best_group
                    for g in selected:
                        total, da, db, lca_val = hop_metrics(devs, nic_topo_addr, g["addr"])
                        assignments.append({
                            "nic_addr": n["addr"],
                            "nic_name": n.get("name") or n["addr"],
                            "nic_driver": n.get("driver"),
                            "nic_vendor": n.get("vendor"),
                            "nic_device": n.get("device"),
                            "nic_numa": n.get("numa"),
                            "nic_linux_name": n.get("linux_name"),
                            "nic_ib_name": n.get("ib_name"),
                            "gpu_addr": g["addr"],
                            "gpu_name": g.get("name") or g["addr"],
                            "gpu_driver": g.get("driver"),
                            "gpu_vendor": g.get("vendor"),
                            "gpu_device": g.get("device"),
                            "gpu_numa": g.get("numa"),
                            "lca": lca_val,
                            "hops_total": total,
                            "nic_up_hops": da,
                            "gpu_up_hops": db,
                            "numa_match": True,  # All in same NUMA
                            "score": None,  # Not used for groups
                            "topology_nic_node": nic_topo_addr if nic_topo_addr != n["addr"] else None,
                            "is_vf": n.get("is_vf", False),
                            "pf_addr": n.get("pf_addr"),
                            "group_max_hops": best_max_hops,
                        })
                    used_gpus.update(g["addr"] for g in selected)

    # Sort assignments: by NIC, then by group_max_hops or score
    assignments.sort(key=lambda r: (r["nic_addr"], r.get("group_max_hops") or r.get("score", (999,999,999,999))[1] if r["score"] else 999))
    return assignments

def filter_devices(devs: Dict[str, Dict], args) -> Dict[str, Dict]:
    """
    Filter devices based on args, and include all ancestors for hierarchy.
    """
    filtered = set()
    for bdf, info in devs.items():
        if is_nic(info) and (not args.nic_vendor or info.get("vendor") in [v.lower() for v in args.nic_vendor]):
            filtered.add(bdf)
        elif (is_gpu_or_display(info) or (args.include_accelerators and is_accelerator(info))) and (not args.gpu_vendor or info.get("vendor") in [v.lower() for v in args.gpu_vendor]):
            filtered.add(bdf)

    # Add ancestors
    to_add = set()
    for bdf in filtered:
        chain = ancestry_chain(devs, bdf)
        to_add.update(chain)
    filtered.update(to_add)

    return {bdf: devs[bdf] for bdf in filtered}

def build_pcie_tree(devs: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Build a mapping from parent BDF to list of child BDFs.
    """
    tree = {}
    for bdf, info in devs.items():
        parent = info.get("parent")
        if parent:
            tree.setdefault(parent, []).append(bdf)
        else:
            tree.setdefault("ROOT", []).append(bdf)
    return tree

def print_pcie_hierarchy(devs: Dict[str, Dict]) -> None:
    """
    Print a visual PCIe hierarchy for all devices.
    """
    tree = build_pcie_tree(devs)

    def print_subtree(bdf, prefix="", is_last=True):
        info = devs.get(bdf, {})
        name = info.get("name") or ""
        vendor = info.get("vendor") or ""
        device = info.get("device") or ""
        cls = info.get("class") or ""
        connector = "└─ " if is_last else "├─ "
        print(f"{prefix}{connector}{bdf} [{vendor}/{device} {cls}] {name}")
        children = tree.get(bdf, [])
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_subtree(child, child_prefix, child_is_last)

    roots = sorted(tree.get("ROOT", []))  # Sort for consistent output
    for i, root in enumerate(roots):
        is_last_root = (i == len(roots) - 1)
        print_subtree(root, "", is_last_root)

def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No NIC?GPU mappings found.")
        return
    headers = [
        "NIC (BDF)", "NIC vendor/device", "NIC drv/numa", "NIC Linux/IB",
        "? GPU (BDF)", "GPU vendor/device", "GPU drv/numa",
        "LCA", "hops (nic?+gpu?=tot)", "group_max_hops"
    ]
    print(" | ".join(headers))
    print("-" * (sum(len(h) for h in headers) + 3*(len(headers)-1)))
    for r in rows:
        nic_meta = f'{(r["nic_vendor"] or "-")}/{(r["nic_device"] or "-")}'
        nic_drv = f'{r["nic_driver"] or "-"} / {r["nic_numa"]}'
        nic_if = f'{r.get("nic_linux_name") or "-"} / {r.get("nic_ib_name") or "-"}'
        gpu_meta = f'{(r["gpu_vendor"] or "-")}/{(r["gpu_device"] or "-")}'
        gpu_drv = f'{r["gpu_driver"] or "-"} / {r["gpu_numa"]}'
        hops = f'{r["nic_up_hops"]}+{r["gpu_up_hops"]}={r["hops_total"]}'
        group_hops = r.get("group_max_hops", "-")
        print(f'{r["nic_addr"]:>12} | {nic_meta:<17} | {nic_drv:<12} | {nic_if:<12} | '
              f'{r["gpu_addr"]:>12} | {gpu_meta:<17} | {gpu_drv:<12} | '
              f'{(r["lca"] or "-"):>12} | {hops} | {group_hops}')

def write_csv(rows: List[Dict], path: str) -> None:
    fields = [
        "nic_addr","nic_name","nic_vendor","nic_device","nic_driver","nic_numa","nic_linux_name","nic_ib_name","is_vf","pf_addr",
        "gpu_addr","gpu_name","gpu_vendor","gpu_device","gpu_driver","gpu_numa",
        "lca","nic_up_hops","gpu_up_hops","hops_total","numa_match","topology_nic_node","group_max_hops"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

def main():
    ap = argparse.ArgumentParser(description="Generic NIC ? GPU/Accelerator PCIe proximity mapper")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="Print JSON instead of table")
    fmt.add_argument("--csv", metavar="FILE", help="Write CSV to file")
    ap.add_argument("--visual", action="store_true", help="Print PCIe hierarchy tree")
    ap.add_argument("--include-accelerators", action="store_true",
                    help="Also consider PCI class 0x12* devices as GPU candidates")
    ap.add_argument("--nic-vendor", action="append", default=[],
                    help="Filter NICs by vendor id (e.g., 0x14e4). Repeatable.")
    ap.add_argument("--gpu-vendor", action="append", default=[],
                    help="Filter GPUs/accels by vendor id (e.g., 0x10de, 0x1002, 0x8086). Repeatable.")
    ap.add_argument("--prefer-same-switch", action="store_true",
                    help="Bias more towards same-switch proximity (lower LCA hops).")
    ap.add_argument("--one-to-one", action="store_true",
                    help="Allocate each GPU to at most one NIC (greedy).")
    ap.add_argument("--max-gpu-per-nic", type=int, default=1,
                    help="Maximum GPUs to assign per NIC (default 1).")
    args = ap.parse_args()

    devs = load_devices()

    if args.visual:
        filtered_devs = filter_devices(devs, args) if (args.nic_vendor or args.gpu_vendor or args.include_accelerators) else devs
        print_pcie_hierarchy(filtered_devs)
        return

    rows = pick_mappings(
        devs,
        nic_vendors=args.nic_vendor or None,
        gpu_vendors=args.gpu_vendor or None,
        include_accels=args.include_accelerators,
        prefer_same_switch=args.prefer_same_switch,
        one_to_one=args.one_to_one,
        max_gpus_per_nic=args.max_gpu_per_nic,
    )

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote {len(rows)} mappings to {args.csv}")
    elif args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
        # Compact summary for quick grep
        print("\n# Compact:")
        for r in rows:
            linux = r.get("nic_linux_name") or "-"
            ib = r.get("nic_ib_name") or "-"
            print(f'{r["nic_addr"]} ({linux}/{ib}) -> {r["gpu_addr"]} | NUMA {r["nic_numa"]}->{r["gpu_numa"]} '
                  f'| hops {r["nic_up_hops"]}+{r["gpu_up_hops"]}={r["hops_total"]} '
                  f'| LCA {r["lca"] or "-"} '
                  f'| {"NUMA-MATCH" if r["numa_match"] else "numa-mismatch"}'
                  + (f' | topo-nic-node {r["topology_nic_node"]}' if r["topology_nic_node"] else "")
                  + (f' | group_max_hops {r["group_max_hops"]}' if r.get("group_max_hops") else ""))

if __name__ == "__main__":
    main()

