import os, json, subprocess

# -------------------------------
# Part 1: Generate cluster.json
# -------------------------------
OUTPUT_CLUSTER_FILE = "./input/cluster.json"
USERNAME = "arravikum"
PRIV_KEY_FILE = "/home/arravikum/.ssh/ara"
HEADNODE = "nova-login-gtu2.prov.zts.cpe.ice.amd.com"
RCCL_FILE = "./input/mi350_config.json"

# Get allocated nodes
nodes = subprocess.check_output(["srun", "hostname"], text=True).splitlines()

cluster = {
    "_comment": "Auto-generated cluster.json from Slurm allocation",
    "username": USERNAME,
    "priv_key_file": PRIV_KEY_FILE,
    "head_node_dict": {"mgmt_ip": HEADNODE},
    "node_dict": {
        node: {"bmc_ip": "NA", "vpc_ip": node}
        for node in nodes
    },
    "bmc_mapping_dict": {},
    "backend_nw_dict": {},
    "frontend_nw_dict": {},
    "storage_dict": {}
}

os.makedirs(os.path.dirname(OUTPUT_CLUSTER_FILE), exist_ok=True)
with open(OUTPUT_CLUSTER_FILE, "w") as f:
    json.dump(cluster, f, indent=4)

print(f"Wrote {OUTPUT_CLUSTER_FILE} with {len(nodes)} nodes")

# -------------------------------
# Part 2: Patch rccl JSON
# -------------------------------
if os.path.exists(RCCL_FILE):
    with open(RCCL_FILE) as f:
        rccl_cfg = json.load(f)

    rccl_cfg["rccl"]["no_of_nodes"] = str(len(nodes))
    rccl_cfg["rccl"]["no_of_global_ranks"] = str(len(nodes) * int(rccl_cfg["rccl"]["ranks_per_node"]))

    with open(RCCL_FILE, "w") as f:
        json.dump(rccl_cfg, f, indent=4)

    print(f"Patched {RCCL_FILE}: no_of_nodes={rccl_cfg['rccl']['no_of_nodes']}, "
          f"no_of_global_ranks={rccl_cfg['rccl']['no_of_global_ranks']}")
else:
    print(f"Warning: {RCCL_FILE} not found, skipping rccl patch")
