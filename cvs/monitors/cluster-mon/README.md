# CVS Cluster Monitor

Real-time GPU cluster monitoring dashboard for AMD Instinct GPUs. One Go
binary owns SSH (including jump host), collectors, RCCL/Inspector, and serves
the existing React UI on `/api`.

## Features

- **Persistent SSH pool** — one TCP+SSH session per node, pessimistic start
  (unreachable until the first probe), keepalives, 5-minute reprobe, jump host
- **GPU metrics** — utilization, temperature, power, memory, PCIe, ECC, XGMI
- **Network** — RDMA links/stats, IP addressing, LLDP topology
- **Software pages** — ROCm / GPU firmware, NIC firmware and drivers, devlink
- **Logs** — AMD dmesg buckets plus validated grep search
- **RCCL** — RAS over SSH local-forward (works through a jump host), Redis
  history for Timeline, Inspector JSONL (file or SSH)
- **Web UI** — Configuration, LLDP install, WebSocket updates

## Quick start (Docker)

Prerequisites: Docker Compose v2, SSH access to the nodes (direct or jump),
`amd-smi` on the nodes.

From this directory (`cvs/monitors/cluster-mon` in the CVS repo):

```bash
cp config/cluster.yaml.example config/cluster.yaml
cp config/nodes.txt.example config/nodes.txt
# edit username, ~/.ssh key_file, and nodes

./full-rebuild.sh
```

Run `./full-rebuild.sh` as the account that owns `~/.ssh`. It sets `CLUSTER_MON_UID`, `CLUSTER_MON_GID`, and `CLUSTER_MON_HOME` from the owner of `config/cluster.yaml`. A direct `docker compose up` does not do that; export those three variables first.

Dashboard: **http://\<host\>:8005** (or `CLUSTER_MON_PORT`).

On first boot, open **Configuration**, set nodes, username, and the key path.
The private key is already on this machine (`ssh.key_file` for direct access,
`jump_host.key_file` when a jump host is used). Compose bind-mounts
`CLUSTER_MON_HOME/.ssh` at the same path in the container. Key paths in `cluster.yaml` are `~/.ssh/<filename>`. Then **Save**. The first metrics sweep waits for
SSH probe; expect a few seconds of empty panels, not a hang.

```bash
sudo docker compose logs -f cvs-cluster-monitor
# look for initial_probe_done reachable:N and metrics_collected gpu_hosts:N
```

### Compose notes

- Host port: `${CLUSTER_MON_PORT:-8005}:8001`
- Config bind-mount: `./config:/app/config`
- SSH keys: `$CLUSTER_MON_HOME/.ssh` bind-mounted read-only at the same path. `cluster.yaml` uses `~/.ssh/<filename>`.
  `./full-rebuild.sh` sets the process uid, gid, and home from the owner of `config/cluster.yaml`. Do not set uid 0.
  Do not mount the git tree.
- Redis: companion container for RCCL snapshot/event history
- Leave `CLUSTER_MON_API_TOKEN` unset unless a reverse proxy injects
  `Authorization: Bearer` or `X-API-Token`. The stock UI does not send it;
  setting the token without a proxy 401s Configuration.

## Configuration

`config/cluster.yaml` and `config/nodes.txt` are gitignored.

```yaml
cluster:
  ssh:
    username: ichristo
    key_file: ~/.ssh/id_ed25519
    timeout: 30
    jump_host:
      enabled: false
      host: jumphost.example.com
      username: jump_user
      key_file: ~/.ssh/id_ed25519
      node_username: ichristo
      node_key_file: ~/.ssh/id_ed25519   # path on the jump host; fetched via SFTP, kept in memory
  polling:
    interval: 60
    failure_threshold: 5
  rccl:
    ras_port: 28028
    poll_interval: 30
```

Passwords entered in the UI stay in process memory and are not written to yaml.

Reload is selective: jump-host identity rebuilds the pool; username/key/password
updates credentials in place; node-list-only uses `Refresh`; polling/RCCL-only
leaves SSH sessions up.

## Development (no Docker)

Go 1.25+ and Node 18+ for a local UI build.

```bash
make ut          # go test ./...
make build       # bin/cluster-mon
make run         # LISTEN=:8001 CONFIG_DIR=./config

# optional static UI
cd frontend && npm ci && npm run build
make run STATIC_DIR=frontend/dist
```

Run Go tests from this directory (`make ut`). The repo-root `make ut` and `make test` targets stay on the Python suite. CI runs `make ut` here as its own step.

## Security

- Do not publish port 8005 on an open campus LAN. Prefer SSH tunnel, VPN, or
  firewall. The UI is HTTP unless you terminate TLS in front.
- Optional `CLUSTER_MON_API_TOKEN` gates `/api/config/*`.
- Jump-host **node** keys are SFTP-fetched and held in memory. They are not on this mount.
- `config/cluster.yaml` and `config/nodes.txt` must not be committed.

## Troubleshooting

```bash
sudo docker logs cvs-cluster-monitor --tail 80
```

| Symptom | What to look for |
|---|---|
| Dashboard lists the node, no GPU data | `pssh_reprobe_failed` … `SSH authentication failed` — wrong user/key. The file in `~/.ssh/` must exist under `CLUSTER_MON_HOME`. |
| `metrics_skipped` `reachable:0` | Probe not done or every host unreachable. Wait for `initial_probe_done`. |
| RCCL `unreachable` then `no_job` | No `rcclras` listener. `no_job` is idle; `unreachable` after a failed start can be leftover Redis state. |
| Key not visible in the container | `CLUSTER_MON_HOME` is not the home that holds `.ssh`. `docker compose down -v` removes Redis only. |
| Configuration 401 | `CLUSTER_MON_API_TOKEN` is set and the browser is not sending it. |

## Architecture

```
React UI  ──HTTP/WS──►  cluster-mon (Go)
                           ├── chi HTTP + gorilla websocket
                           ├── internal/pssh  (jump-capable pool)
                           ├── collectors (gpu / nic / logs / software)
                           ├── rccl + inspector
                           └── Redis (optional RCCL history)
```

The image and `cmd/cluster-mon` are the monitor. There is no Python backend
or UDS `gpu-collector` in this tree.

## License

MIT License — see the repository LICENSE.
