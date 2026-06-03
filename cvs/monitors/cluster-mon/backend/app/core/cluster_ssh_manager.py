"""
ClusterSshManager: cluster-mon's SSH manager backed by cvs.lib.parallel.

Thin adapter that preserves the exact API cluster-mon already depends on
(async ``exec_async``; sync ``exec`` / ``exec_cmd_list``; ``get_*_hosts``;
``refresh_host_reachability`` / ``recreate_client`` / ``destroy_clients``; and
the ``host_list`` / ``reachable_hosts`` / ``unreachable_hosts`` attributes) while
delegating the actual SSH work to ``cvs.lib.parallel.multiprocess_pssh.MultiProcessPssh``.

It replaces the two legacy classes (``cvs_parallel_ssh_reliable.Pssh`` and
``jump_host_pssh.JumpHostPssh``). Behavioral parity that does NOT come for free
from ``MultiProcessPssh`` is implemented explicitly here and pinned by
``app/core/unittests/test_ssh_manager_contract.py``:

- TCP pre-probe (direct path only) prunes dead hosts before SSH is attempted.
- Hosts dropped by the pre-probe never reach ``MultiProcessPssh``, so their
  ``ABORT: Host Unreachable Error`` marker is merged back into the result dict
  (mirrors ``Pssh.inform_unreachability``).
- When there are no reachable hosts, every original host gets the ABORT marker
  (mirrors ``Pssh.exec`` when its client is ``None``).
- ``exec_async`` offloads the blocking ``exec`` via ``asyncio.to_thread`` and
  serializes calls with an ``asyncio.Lock`` (mirrors the old ``_ssh_lock``).
"""

import asyncio
import logging
import threading

from cvs.lib.parallel.config import ParallelConfig
from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh

from app.core.host_probe import discover_reachable_hosts

logger = logging.getLogger(__name__)

ABORT_MARKER = "ABORT: Host Unreachable Error"

# TCP pre-probe tuning (matches the legacy Pssh defaults).
_PROBE_PORT = 22
_PROBE_TIMEOUT = 5
_PROBE_MAX_WORKERS = 100


class ClusterSshManager:
    """Async-friendly SSH manager wrapping ``MultiProcessPssh``.

    Args mirror what cluster-mon's ``main.py`` passes today. When ``jump_host``
    is set the lib's native libssh2 ``proxy_*`` tunnel is used and the TCP
    pre-probe is skipped (it cannot traverse the bastion); reachability is then
    pruned lazily by the lib through the tunnel.
    """

    def __init__(
        self,
        host_list,
        user=None,
        password=None,
        pkey='id_rsa',
        timeout=30,
        stop_on_errors=False,
        jump_host=None,
        jump_user=None,
        jump_password=None,
        jump_pkey=None,
        jump_port=22,
    ):
        self.host_list = list(host_list or [])
        self.user = user
        self.password = password
        self.pkey = pkey
        self.timeout = timeout
        self.stop_on_errors = stop_on_errors

        self._jump = {
            'jump_host': jump_host,
            'jump_user': jump_user,
            'jump_password': jump_password,
            'jump_pkey': jump_pkey,
            'jump_port': jump_port,
        }
        # Direct path uses the TCP pre-probe; jump-host path relies on the lib's
        # lazy prune through the tunnel (a TCP probe can't reach nodes directly).
        self._use_preprobe = jump_host is None

        # threading.Lock protects the (not thread-safe) underlying SSH stack for
        # sync callers invoked from worker threads (collectors use
        # asyncio.to_thread). The asyncio.Lock that serializes exec_async is
        # created lazily on first use so it binds to the running event loop
        # rather than whatever loop happened to exist at construction time.
        self._async_lock = None
        self._exec_lock = threading.Lock()

        if self._use_preprobe:
            logger.info("Probing %d hosts for reachability...", len(self.host_list))
            self.reachable_hosts, self.unreachable_hosts = discover_reachable_hosts(
                self.host_list, port=_PROBE_PORT, timeout=_PROBE_TIMEOUT, max_workers=_PROBE_MAX_WORKERS
            )
            logger.info(
                "Probe completed: %d reachable, %d unreachable",
                len(self.reachable_hosts),
                len(self.unreachable_hosts),
            )
        else:
            logger.info("Jump host configured (%s) - skipping TCP pre-probe", jump_host)
            self.reachable_hosts = list(self.host_list)
            self.unreachable_hosts = []

        self._mp = self._build_manager() if self.reachable_hosts else None

    def _build_manager(self):
        """Construct a ``MultiProcessPssh`` over the current reachable hosts."""
        config = ParallelConfig.from_env()
        # Reuse shard worker processes across operations to avoid per-call spawn
        # storms during the metrics-collection loop.
        config.persistent_shards = True
        if self._jump['jump_host']:
            config.jump_host = self._jump['jump_host']
            config.jump_user = self._jump['jump_user']
            config.jump_password = self._jump['jump_password']
            config.jump_pkey = self._jump['jump_pkey']
            config.jump_port = self._jump['jump_port']

        return MultiProcessPssh(
            logger,
            self.reachable_hosts,
            user=self.user,
            password=self.password,
            pkey=self.pkey,
            stop_on_errors=self.stop_on_errors,
            config=config,
        )

    def _merge_unreachable(self, cmd_output):
        """Append the ABORT marker for every pre-probe-unreachable host.

        Mirrors ``Pssh.inform_unreachability``: hosts dropped by the TCP
        pre-probe never reached the SSH layer, so the consumer-facing result
        dict must still surface them as failures.
        """
        for host in self.unreachable_hosts:
            cmd_output[host] = cmd_output.get(host, "") + "\n" + ABORT_MARKER
        return cmd_output

    def _all_abort(self):
        """Result when there are no reachable hosts (mirrors Pssh.exec client=None)."""
        return {host: ABORT_MARKER for host in self.host_list}

    def exec(self, cmd, timeout=None, print_console=True):
        """Run ``cmd`` on all reachable hosts; returns ``{host: output_str}``."""
        if self._mp is None:
            logger.warning("No reachable hosts - returning ABORT for all hosts")
            return self._all_abort()
        with self._exec_lock:
            cmd_output = self._mp.exec(cmd, timeout, print_console)
        return self._merge_unreachable(cmd_output)

    def exec_cmd_list(self, cmd_list, timeout=None, print_console=True):
        """Run a per-host command list; returns ``{host: output_str}``."""
        if self._mp is None:
            logger.warning("No reachable hosts - returning ABORT for all hosts")
            return self._all_abort()
        with self._exec_lock:
            cmd_output = self._mp.exec_cmd_list(cmd_list, timeout, print_console)
        return self._merge_unreachable(cmd_output)

    async def exec_async(self, cmd, timeout=None, print_console=True):
        """Async wrapper: offloads the blocking ``exec`` to a worker thread.

        Serialized with an ``asyncio.Lock`` so concurrent collectors don't
        interleave SSH operations, while keeping the event loop responsive.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            return await asyncio.to_thread(self.exec, cmd, timeout, print_console)

    def get_reachable_hosts(self):
        """Return a copy of the reachable host list."""
        return list(self.reachable_hosts)

    def get_unreachable_hosts(self):
        """Return a copy of the unreachable host list."""
        return list(self.unreachable_hosts)

    def refresh_host_reachability(self):
        """Re-probe the original host list; return True if reachability changed.

        Direct path only. Under a jump host the TCP probe can't reach nodes, so
        this is a no-op (the lib prunes lazily through the tunnel) and returns
        False.
        """
        if not self._use_preprobe:
            logger.debug("Jump host active - skipping TCP re-probe (lazy prune handles it)")
            return False

        logger.info("Refreshing host reachability...")
        old_reachable = set(self.reachable_hosts)
        new_reachable, new_unreachable = discover_reachable_hosts(
            self.host_list, port=_PROBE_PORT, timeout=_PROBE_TIMEOUT, max_workers=_PROBE_MAX_WORKERS
        )
        self.reachable_hosts = new_reachable
        self.unreachable_hosts = new_unreachable
        return set(new_reachable) != old_reachable

    def recreate_client(self):
        """Rebuild the underlying ``MultiProcessPssh`` from current reachable hosts."""
        if self._mp is not None:
            try:
                self._mp.destroy_clients()
            except Exception:
                logger.debug("Ignoring error while destroying previous SSH manager", exc_info=True)

        if not self.reachable_hosts:
            logger.warning("No reachable hosts - clearing SSH manager")
            self._mp = None
            return

        logger.info("Recreating SSH manager with %d reachable hosts...", len(self.reachable_hosts))
        self._mp = self._build_manager()

    def destroy_clients(self):
        """Tear down the underlying SSH manager and its worker processes."""
        if self._mp is not None:
            self._mp.destroy_clients()
            self._mp = None
