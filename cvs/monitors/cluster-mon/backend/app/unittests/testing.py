"""
Shared test helpers for the cluster-mon backend test suite.

`FakeSshManager` is an in-memory stand-in for the SSH manager (currently
`cvs_parallel_ssh_reliable.Pssh` / `jump_host_pssh.JumpHostPssh`, later the
`ClusterSshManager` adapter). It implements the exact surface the collectors and
API endpoints depend on so tests can pin the `exec_async(cmd) -> {host: str}`
contract without any real SSH or network access.
"""

from typing import Dict, List, Optional


class FakeSshManager:
    """
    Test double mirroring the cluster-mon SSH-manager API.

    Responses are configured per command via ``command_map``: a mapping of
    "command substring" -> {host: output_str}. On ``exec_async`` the first
    substring found in the command wins (insertion order); if none match, every
    reachable host gets ``default_output``. This lets a single fake serve the
    multiple distinct commands a collector issues in one collection pass.
    """

    def __init__(
        self,
        reachable_hosts: List[str],
        unreachable_hosts: Optional[List[str]] = None,
        command_map: Optional[Dict[str, Dict[str, str]]] = None,
        default_output: str = "",
        refresh_changed: bool = False,
        cmd_list_response: Optional[Dict[str, str]] = None,
    ):
        self.host_list = list(reachable_hosts) + list(unreachable_hosts or [])
        self.reachable_hosts = list(reachable_hosts)
        self.unreachable_hosts = list(unreachable_hosts or [])
        self._command_map = dict(command_map or {})
        self.default_output = default_output
        self._refresh_changed = refresh_changed
        self._cmd_list_response = dict(cmd_list_response) if cmd_list_response is not None else None
        # Recorded {"cmd", "timeout"} for each exec/exec_async call, for assertions.
        self.calls: List[Dict[str, object]] = []
        self.cmd_list_calls: List[Dict[str, object]] = []
        self.recreate_calls = 0
        self.destroy_calls = 0

    def set_response(self, command_substring: str, host_output_map: Dict[str, str]) -> None:
        self._command_map[command_substring] = dict(host_output_map)

    def _resolve(self, cmd: str) -> Dict[str, str]:
        for needle, mapping in self._command_map.items():
            if needle in cmd:
                return dict(mapping)
        return {host: self.default_output for host in self.reachable_hosts}

    def exec(self, cmd: str, timeout: Optional[int] = None, print_console: bool = True) -> Dict[str, str]:
        self.calls.append({"cmd": cmd, "timeout": timeout})
        return self._resolve(cmd)

    async def exec_async(self, cmd: str, timeout: Optional[int] = None, print_console: bool = True) -> Dict[str, str]:
        self.calls.append({"cmd": cmd, "timeout": timeout})
        return self._resolve(cmd)

    def exec_cmd_list(
        self, cmd_list: List[str], timeout: Optional[int] = None, print_console: bool = True
    ) -> Dict[str, str]:
        self.cmd_list_calls.append({"cmd_list": list(cmd_list), "timeout": timeout})
        if self._cmd_list_response is not None:
            return dict(self._cmd_list_response)
        return {host: self.default_output for host in self.reachable_hosts}

    def get_reachable_hosts(self) -> List[str]:
        return list(self.reachable_hosts)

    def get_unreachable_hosts(self) -> List[str]:
        return list(self.unreachable_hosts)

    def refresh_host_reachability(self) -> bool:
        return self._refresh_changed

    def recreate_client(self) -> None:
        self.recreate_calls += 1

    def destroy_clients(self) -> None:
        self.destroy_calls += 1
