"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import logging
from typing import Optional

from cvs.core.scope import ExecResult, ExecScope
from cvs.lib.parallel_ssh_lib import Pssh


class PsshTransport:
    """Transport implementation that wraps the existing parallel-ssh Pssh helper.

    Holds two underlying Pssh handles:
      * _all  - one client across every reachable host (ExecScope.ALL)
      * _head - one client across just the head node     (ExecScope.HEAD)

    For ExecScope.SUBSET the transport spins up an ad hoc Pssh for just the
    requested subset. Subsets are expected to be rare (debugging / per-host
    probes); the common scopes use the long-lived clients.

    All exec results are returned as dict[str, ExecResult]. The structured
    Pssh.exec(detailed=True) contract is consumed internally and translated;
    callers of the Transport API never see Pssh's two return shapes.
    """

    def __init__(
        self,
        hosts: list[str],
        head_node: str,
        username: str,
        priv_key_file: str,
        password: Optional[str] = None,
        env_vars: Optional[dict] = None,
        log: Optional[logging.Logger] = None,
        stop_on_errors: bool = False,
    ):
        self.hosts = list(hosts)
        self.head_node = head_node
        self.log = log or logging.getLogger(__name__)
        self._username = username
        self._priv_key_file = priv_key_file
        self._password = password
        self._env_vars = env_vars or {}
        self._stop_on_errors = stop_on_errors

        self._all = self._build_pssh(self.hosts)
        self._head = self._build_pssh([self.head_node])

        # env_prefix is exposed for consumers that want to see what got threaded
        # through to the underlying Pssh (e.g. tests asserting Bug 5 stays fixed).
        self.env_prefix = self._all.env_prefix

    def _build_pssh(self, hosts: list[str]) -> Pssh:
        return Pssh(
            self.log,
            hosts,
            user=self._username,
            password=self._password,
            pkey=self._priv_key_file,
            host_key_check=False,
            stop_on_errors=self._stop_on_errors,
            env_vars=self._env_vars,
        )

    def _client_for(self, scope: ExecScope, subset: Optional[list[str]]) -> Pssh:
        if scope is ExecScope.ALL:
            return self._all
        if scope is ExecScope.HEAD:
            return self._head
        if scope is ExecScope.SUBSET:
            if not subset:
                raise ValueError("ExecScope.SUBSET requires a non-empty `subset` list")
            return self._build_pssh(subset)
        raise ValueError(f"unknown ExecScope: {scope}")

    @staticmethod
    def _to_results(raw: dict) -> dict[str, ExecResult]:
        """Translate Pssh.exec(detailed=True) shape into dict[str, ExecResult]."""
        results: dict[str, ExecResult] = {}
        for host, payload in raw.items():
            if isinstance(payload, dict):
                output = payload.get("output", "")
                exit_code = payload.get("exit_code", -1)
            else:
                # Defensive: should not happen because we always pass detailed=True.
                output = str(payload)
                exit_code = -1
            results[host] = ExecResult(host=host, output=output, exit_code=exit_code)
        return results

    def exec(
        self,
        cmd: str,
        scope: ExecScope,
        *,
        subset: Optional[list[str]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, ExecResult]:
        client = self._client_for(scope, subset)
        raw = client.exec(cmd, timeout=timeout, detailed=True)
        return self._to_results(raw)

    def scp(
        self,
        src: str,
        dst: str,
        scope: ExecScope,
        *,
        subset: Optional[list[str]] = None,
    ) -> None:
        client = self._client_for(scope, subset)
        client.scp_file(src, dst)
