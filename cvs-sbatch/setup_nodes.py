import subprocess
import paramiko
import sys
import os
import json
import re

# Commands to execute remotely before health checks
base_commands = ["sudo modprobe amdgpu", "sudo /apps/shared/disable_acs.sh", "sudo sysctl kernel.numa_balancing=0"]


def execute_on_host(hostname, username="arravikum", key_path="~/.ssh/id_rsa"):
    """Run setup and health checks on a remote host. Returns True if all checks pass."""
    print(f"\nConnecting to {hostname} ...")
    key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    node_passed = True

    try:
        ssh.connect(hostname, username=username, pkey=key)

        # 1. Run setup commands
        for cmd in base_commands:
            print(f"[{hostname}]$ {cmd}")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            out, err = stdout.read().decode().strip(), stderr.read().decode().strip()
            if out:
                print(f"{hostname}: {out}")
            if err:
                print(f"{hostname} [ERR]: {err}")

        # 2. Check for running Docker containers
        print(f"[{hostname}] Checking for running Docker containers ...")
        docker_check_cmd = "sudo docker ps -q"
        stdin, stdout, stderr = ssh.exec_command(docker_check_cmd)
        container_ids = stdout.read().decode().strip().splitlines()

        if container_ids:
            print(f"{hostname}: Found {len(container_ids)} running Docker container(s). Stopping and removing them...")
            stop_cmd = "sudo docker stop $(sudo docker ps -q) && sudo docker rm $(sudo docker ps -a -q)"
            stdin, stdout, stderr = ssh.exec_command(stop_cmd)
            out, err = stdout.read().decode().strip(), stderr.read().decode().strip()
            if out:
                print(f"{hostname}: {out}")
            if err:
                print(f"{hostname} [ERR]: {err}")
        else:
            print(f"{hostname}: No running Docker containers found.")
        # 3. Verify AMDGPU driver loaded
        stdin, stdout, stderr = ssh.exec_command(
            "lsmod | grep -q amdgpu && echo 'amdgpu loaded' || echo 'amdgpu not loaded'"
        )
        driver_status = stdout.read().decode().strip()
        print(f"{hostname}: Driver check -> {driver_status}")
        if "not loaded" in driver_status:
            print(f"{hostname}: amdgpu driver not loaded")
            node_passed = False

        # 4. Check NICs using ip -o link show
        stdin, stdout, stderr = ssh.exec_command("ip -o link show")
        nic_output = stdout.read().decode().strip().splitlines()
        nics_up, nics_down = [], []

        for line in nic_output:
            match = re.match(r"\d+:\s+([^:]+):\s+<([^>]+)>.*state\s+(\S+)", line)
            if match:
                iface, flags, state = match.groups()
                if "UP" in flags.split(",") or state == "UP":
                    nics_up.append(iface)
                else:
                    nics_down.append(iface)

        if nics_up:
            print(f"{hostname}: NICs UP: {', '.join(nics_up)}")
        else:
            print(f"{hostname}: No NICs appear to be UP")
            node_passed = False

        if nics_down:
            print(f"{hostname}: NICs DOWN: {', '.join(nics_down)}")

        # 5. Killing stale processes on GPU nodes before script launch
        print(f"[{hostname}] Cleaning up stale GPU processes (amd-smi) ...")
        cmd_new = "kill -9 $(amd-smi process | grep PID | awk '{print $2}')"
        stdin, stdout, stderr = ssh.exec_command(cmd_new)
        # Verify amd-smi is available (via sudo PATH)
        check_amd_smi_cmd = "sudo command -v amd-smi >/dev/null 2>&1 && echo OK || echo MISSING"
        stdin, stdout, stderr = ssh.exec_command(check_amd_smi_cmd)
        amd_smi_status = stdout.read().decode().strip()

        if amd_smi_status != "OK":
            print(f"{hostname} [WARN]: amd-smi not found or not accessible via sudo, skipping GPU process cleanup")
        else:
            # Extract GPU PIDs safely
            list_pids_cmd = "sudo amd-smi process | awk '/PID/ {print $2}' | grep -E '^[0-9]+$' || true"
            stdin, stdout, stderr = ssh.exec_command(list_pids_cmd)
            pids = stdout.read().decode().split()

            if not pids:
                print(f"{hostname}: No active GPU processes found")
            else:
                print(f"{hostname}: Found GPU PIDs: {', '.join(pids)}")

                # Try graceful termination first
                term_cmd = f"sudo kill {' '.join(pids)}"
                ssh.exec_command(term_cmd)

                # Give processes a moment to exit
                ssh.exec_command("sleep 2")

                # Check which PIDs are still alive
                kill_check_cmd = "ps -o pid= -p " + " ".join(pids) + " 2>/dev/null | tr -d ' '"
                stdin, stdout, stderr = ssh.exec_command(kill_check_cmd)
                remaining = stdout.read().decode().split()

                if remaining:
                    print(f"{hostname}: Forcing kill on remaining PIDs: {', '.join(remaining)}")
                    ssh.exec_command(f"sudo kill -9 {' '.join(remaining)}")
                else:
                    print(f"{hostname}: GPU processes terminated cleanly")

        # 6. Check GPU count using rocm-smi JSON
        gpu_cmd = "sudo rocm-smi --showbus --json 2>/dev/null"
        stdin, stdout, stderr = ssh.exec_command(gpu_cmd)
        gpu_json = stdout.read().decode().strip()

        try:
            gpu_info = json.loads(gpu_json)
            gpu_count = len(gpu_info)
            if gpu_count == 8:
                print(f"{hostname}: Detected {gpu_count} GPUs")
            else:
                print(f"{hostname}: Expected 8 GPUs, found {gpu_count}")
                node_passed = False
        except json.JSONDecodeError:
            print(f"{hostname}: Failed to parse rocm-smi JSON output")
            node_passed = False
            print(f"{hostname}: Raw output:\n{gpu_json}")

        ssh.close()
        if node_passed:
            print(f"{hostname}: All checks PASSED\n")
        else:
            print(f"{hostname}: One or more checks FAILED\n")

        return node_passed

    except Exception as e:
        print(f"Failed to connect to {hostname}: {e}")
        return False


# --- Main section ---

try:
    print("Fetching hostnames via srun ...")
    hosts_output = subprocess.check_output(["srun", "hostname"], text=True)
    hosts = sorted(set(hosts_output.strip().splitlines()))
except subprocess.CalledProcessError as e:
    print(f"Error running srun hostname: {e}")
    sys.exit(1)

print(f"Found {len(hosts)} hosts: {', '.join(hosts)}")

any_failed = False
for host in hosts:
    success = execute_on_host(host)
    if not success:
        any_failed = True

if any_failed:
    print("One or more hosts failed checks. Exiting with non-zero status.")
    sys.exit(1)
else:
    print("All hosts passed health checks.")
    sys.exit(0)
