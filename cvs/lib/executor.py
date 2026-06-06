"""Executor wrappers around the project's MultiProcessPssh.

Callers expect this contract (used by base_adapter, vllm adapter, cli/run.py,
resource_resolver):
- ``executor.exec(cmd, timeout=...)`` returns a stdout-string and raises
  ``RuntimeError`` on non-zero exit. (Single-host behaviour.)
- ``MultiHostExecutor.executor_for(host)`` returns a per-host executor with
  the same shape.

Backed by ``MultiProcessPssh``: keepalive, shared connections, optional
sharding, structured per-host output. The structured output is the root-cause
fix for the FLAT=/HF=/sha256: sentinel scanning workarounds that resource
resolver previously needed — pssh separates per-host stdout from local ssh
banner noise.
"""

from __future__ import annotations

import subprocess

from cvs.lib.parallel.multiprocess_pssh import MultiProcessPssh


def _make_pssh(hosts: list[str], *, user: str, priv_key: str | None,
               env_vars: dict[str, str] | None) -> MultiProcessPssh:
    return MultiProcessPssh(
        None,                              # log — deprecated positional
        list(hosts),
        user=user,
        pkey=priv_key or 'id_rsa',
        host_key_check=False,
        stop_on_errors=False,
        env_vars=env_vars or None,
    )


class _PsshHostView:
    """Per-host adapter: wraps a single-host MultiProcessPssh."""

    def __init__(self, host: str, *, user: str, priv_key: str | None,
                 env_vars: dict[str, str] | None) -> None:
        self.host = host
        self._mp = _make_pssh([host], user=user, priv_key=priv_key, env_vars=env_vars)

    def exec(self, cmd: str, timeout: float | None = None) -> str:
        out = self._mp.exec(cmd, timeout=timeout, print_console=False, detailed=True)
        entry = out.get(self.host)
        if entry is None:
            raise RuntimeError(f"pssh: no result for host {self.host!r} (got {list(out)})")
        body = entry["output"]
        rc = entry["exit_code"]
        if rc != 0:
            raise RuntimeError(f"ssh {self.host} rc={rc}: {body.strip()[:500]}")
        return body


class LocalExecutor:
    """Run on the local host (for tests / when CVS runs on the target node)."""

    def exec(self, cmd: str, timeout: float | None = None) -> str:
        r = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            raise RuntimeError(f"local rc={r.returncode}: {out.strip()[:500]}")
        return out


class MultiHostExecutor:
    """Wraps MultiProcessPssh. ``exec`` broadcasts; ``executor_for`` returns a per-host view."""

    def __init__(
        self,
        hosts: list[str],
        *,
        user: str,
        priv_key: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self.hosts = list(hosts)
        self._user = user
        self._priv_key = priv_key
        self._env_vars = env_vars
        self._mp = _make_pssh(self.hosts, user=user, priv_key=priv_key, env_vars=env_vars)
        self._views = {
            h: _PsshHostView(h, user=user, priv_key=priv_key, env_vars=env_vars)
            for h in self.hosts
        }

    def executor_for(self, host: str) -> _PsshHostView:
        if host not in self._views:
            raise KeyError(f"host {host!r} not in executor pool {list(self._views)}")
        return self._views[host]

    def exec(self, cmd: str, timeout: float | None = None) -> dict[str, str]:
        """Broadcast — returns {host: stdout}. Raises if ANY host non-zero."""
        out = self._mp.exec(cmd, timeout=timeout, print_console=False, detailed=True)
        bad = {h: v for h, v in out.items() if v["exit_code"] != 0}
        if bad:
            msgs = "; ".join(f"{h} rc={v['exit_code']}" for h, v in bad.items())
            raise RuntimeError(f"pssh broadcast failures: {msgs}")
        return {h: v["output"] for h, v in out.items()}
