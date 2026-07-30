import json
import logging
import sys
import os
import warnings

from .base import SubcommandPlugin
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.globals import set_log_level


def _collect_switch_hosts(cluster):
    """
    Return a flat list of all switch tray IPs across all racks.

    Walks the 'racks' (or deprecated 'rack_groups') block and collects
    every IP listed under switch_trays. Credential resolution is NOT done
    here — the caller resolves the single fleet-wide switch credential from
    the racks block.
    """
    racks_raw = cluster.get('racks') or cluster.get('rack_groups', {})
    if not isinstance(racks_raw, dict):
        return []
    skip = {'switch_ssh_user', 'switch_ssh_password', 'switch_ssh_key_file'}
    hosts = []
    for rack_id, rack in racks_raw.items():
        if rack_id in skip or not isinstance(rack, dict):
            continue
        hosts.extend(rack.get('switch_trays', []))
    return hosts


class ExecPlugin(SubcommandPlugin):
    def get_name(self):
        return "exec"

    def get_parser(self, subparsers):
        parser = subparsers.add_parser("exec", help="Execute a command on all nodes in the cluster")
        parser.add_argument("--cmd", required=True, help="Command to execute on all nodes")
        parser.add_argument(
            "--cluster_file",
            help="Path to cluster configuration JSON file (takes precedence over CLUSTER_FILE env var)",
        )
        parser.add_argument(
            "--target",
            choices=["computes", "switches", "all"],
            default="computes",
            help=(
                "Scope of execution: "
                "'computes' (default) runs on all node_dict hosts; "
                "'switches' runs on all switch_trays from the racks block; "
                "'all' runs on both."
            ),
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="Per-node command output timeout in seconds (default: 30)",
        )
        parser.add_argument(
            "--connect-timeout",
            type=int,
            default=15,
            dest="connect_timeout",
            help=(
                "Per-node SSH connection timeout in seconds (default: 15). "
                "Unreachable hosts fail after this many seconds regardless of --timeout."
            ),
        )
        parser.add_argument(
            "--json",
            action="store_true",
            default=False,
            dest="json_output",
            help=(
                "Emit output as a JSON object instead of human-readable text. "
                "Schema: {command, read_timeout, connect_timeout, output: {host: str}}."
            ),
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            default=False,
            help=(
                "Show internal SSH connection diagnostics (SocketDisconnectError, "
                "AuthenticationError, Timeout, pruning messages). Suppressed by default."
            ),
        )
        parser.set_defaults(_plugin=self)
        return parser

    def get_epilog(self):
        return """
Exec Commands:
  cvs exec --cmd "hostname" --cluster_file cluster.json              Execute on all compute nodes (default)
  cvs exec --cmd "show version" --cluster_file cluster.json --target switches   Execute on switch trays
  cvs exec --cmd "date" --cluster_file cluster.json --target all     Execute on computes + switches
  cvs exec --cmd "long_job" --timeout 300 --connect-timeout 10       Long command, fast fail on unreachable hosts
  cvs exec --cmd "hostname" --json | jq '.output'                    JSON output, pipe to jq
  cvs exec --cmd "hostname" --verbose                                Show SSH diagnostics (suppressed by default)
  CLUSTER_FILE=cluster.json cvs exec --cmd "hostname"                Use env var for cluster file"""

    def _run_on_hosts(
        self,
        hosts,
        username,
        env_vars,
        cmd,
        *,
        label,
        pkey=None,
        password=None,
        timeout=None,
        connect_timeout=None,
    ):
        """
        Run cmd on the given host list. Returns (success, host_output_dict).

        Args:
            timeout:         Per-node command output read timeout (seconds).
                             Passed as read_timeout to pssh.exec(). Controls how
                             long to wait for each host's stdout after the command
                             starts running.
            connect_timeout: Per-node SSH connection timeout (seconds).
                             Passed to ParallelSSHClient via Pssh constructor.
                             Unreachable or slow-to-handshake hosts will fail
                             after this many seconds, independently of timeout.

        Returns:
            (bool, dict[str, str]): success flag and mapping of host -> output string.
            On SSH init or exec failure, returns (False, {}).
        """
        log = logging.getLogger(__name__)
        try:
            pssh = Pssh(
                log=log,
                host_list=hosts,
                user=username,
                pkey=pkey,
                password=password,
                stop_on_errors=False,
                env_vars=env_vars,
                num_retries=0,
                # connect_timeout bounds TCP/SSH handshake per host; without this
                # unreachable hosts hang for the OS default (~60-120s) regardless
                # of the command timeout.
                **(dict(timeout=connect_timeout) if connect_timeout is not None else {}),
            )
        except Exception as e:
            print(f"Error initializing SSH for {label}: {e}", file=sys.stderr)
            return False, {}

        try:
            output = pssh.exec(cmd, timeout=timeout)
        except Exception as e:
            print(f"Error executing command on {label}: {e}", file=sys.stderr)
            return False, {}

        return True, output

    def _emit_error(self, msg, json_mode):
        """Print an error message. In JSON mode, writes to stderr so stdout stays valid JSON."""
        print(msg, file=sys.stderr if json_mode else sys.stdout)

    def _print_text_output(self, label, host_output):
        """Print host results in human-readable format."""
        for host, out in host_output.items():
            print(f"[{label}] Host: {host}")
            print(out)
            print("---")

    def run(self, args):
        json_mode = getattr(args, 'json_output', False)
        verbose = getattr(args, 'verbose', False)

        # Suppress WARNING-level SSH/pssh diagnostic noise by default.
        # With --verbose, leave the root logger at its configured level so all
        # messages (SocketDisconnectError, AuthenticationError, pruning, etc.) show.
        if not verbose:
            set_log_level(logging.ERROR)

        # CLI flag wins; env var is the fallback.
        cluster_file = args.cluster_file or os.environ.get('CLUSTER_FILE')
        if not cluster_file:
            self._emit_error(
                "Error: No cluster file specified. Set CLUSTER_FILE environment variable or use --cluster_file.",
                json_mode,
            )
            sys.exit(1)

        try:
            with open(cluster_file, 'r') as f:
                cluster = json.load(f)
        except FileNotFoundError:
            self._emit_error(f"Error: Cluster file '{cluster_file}' not found.", json_mode)
            sys.exit(1)
        except json.JSONDecodeError as e:
            self._emit_error(f"Error: Invalid JSON in cluster file: {e}", json_mode)
            sys.exit(1)

        username = cluster.get('username')
        if not username:
            self._emit_error("Error: 'username' not found in cluster file.", json_mode)
            sys.exit(1)

        env_vars = cluster.get('env_vars')
        target = args.target
        overall_failed = False
        combined_output = {}  # accumulated for --json mode

        # --- Compute trays ---
        if target in ('computes', 'all'):
            pkey = cluster.get('priv_key_file')
            if not pkey:
                self._emit_error(
                    "Error: 'priv_key_file' not found in cluster file (required for compute targets).",
                    json_mode,
                )
                sys.exit(1)
            node_dict = cluster.get('node_dict', {})
            compute_hosts = list(node_dict.keys())
            if not compute_hosts:
                self._emit_error("Error: No hosts found in node_dict.", json_mode)
                sys.exit(1)
            ok, host_output = self._run_on_hosts(
                compute_hosts,
                username,
                env_vars,
                args.cmd,
                label="compute",
                pkey=pkey,
                timeout=args.timeout,
                connect_timeout=args.connect_timeout,
            )
            if not ok:
                overall_failed = True
            if json_mode:
                combined_output.update(host_output)
            else:
                self._print_text_output("compute", host_output)

        # --- Switch trays ---
        if target in ('switches', 'all'):
            # Emit deprecation warning if the old rack_groups top-level key is used.
            if cluster.get('rack_groups') is not None and cluster.get('racks') is None:
                warnings.warn(
                    "'rack_groups' in cluster.json is deprecated. Rename it to 'racks'.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            racks_raw = cluster.get('racks') or cluster.get('rack_groups', {})
            switch_hosts = _collect_switch_hosts(cluster)
            if not switch_hosts:
                self._emit_error(
                    "Warning: --target includes switches but no switch_trays found in 'racks' block.",
                    json_mode,
                )
            else:
                sw_user = racks_raw.get('switch_ssh_user') if isinstance(racks_raw, dict) else None
                sw_key = racks_raw.get('switch_ssh_key_file') if isinstance(racks_raw, dict) else None
                sw_pw = (
                    None if sw_key else (racks_raw.get('switch_ssh_password') if isinstance(racks_raw, dict) else None)
                )
                ok, host_output = self._run_on_hosts(
                    switch_hosts,
                    sw_user,
                    env_vars,
                    args.cmd,
                    label="switch",
                    pkey=sw_key,
                    password=sw_pw,
                    timeout=args.timeout,
                    connect_timeout=args.connect_timeout,
                )
                if not ok:
                    overall_failed = True
                if json_mode:
                    combined_output.update(host_output)
                else:
                    self._print_text_output("switch", host_output)

        if json_mode:
            envelope = {
                "command": args.cmd,
                "read_timeout": args.timeout,
                "connect_timeout": args.connect_timeout,
                "output": combined_output,
            }
            print(json.dumps(envelope, indent=2))

        if overall_failed:
            sys.exit(1)
