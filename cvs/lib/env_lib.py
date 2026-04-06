"""
env_lib.py

Utilities for safely constructing shell-compatible environment variable
exports, with controlled support for self-referential expansion (e.g. PATH).

Key guarantees:
- Prevents shell injection by quoting all user-provided values.
- Allows controlled expansion ONLY for `$KEY` where KEY is the variable being
  assigned (e.g., PATH=/x:$PATH, LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/y).
- Expansion is performed on the *remote shell*, not locally.
"""

import shlex


def build_env_prefix(env_vars):
    """
    Build a shell-safe export prefix from environment variables.

    Supported patterns:
      - <prefix>:$VAR   (prepend)
      - $VAR:<suffix>   (append)

    Unsupported patterns (treated as literal values):
      - Cross-variable expansion (e.g., FOO=$BAR)
      - Shell substitution (e.g., $(...), `...`)
      - Parameter expansion (e.g., ${VAR:-default})

    Args:
        env_vars: Dictionary of environment variables to export.

    Returns:
        A string suitable for prefixing a shell command, e.g.:
            "export PATH=/x:$PATH ; export FOO='bar'"
        or an empty string if env_vars is empty.
    """
    if not env_vars:
        return ""

    exports = []

    for key, value in env_vars.items():
        marker = f"${key}"

        # Case 1: Prepend to existing variable (e.g., PATH=/x:$PATH)
        if value.endswith(":" + marker):
            prefix = value[: -(len(marker) + 1)]
            exports.append(f"export {key}={shlex.quote(prefix)}:${key}")

        # Case 2: Append to existing variable (e.g., PATH=$PATH:/x)
        elif value.startswith(marker + ":"):
            suffix = value[len(marker) + 1 :]
            exports.append(f"export {key}=${key}:{shlex.quote(suffix)}")

        # Case 3: Treat as a literal value (fully quoted)
        else:
            exports.append(f"export {key}={shlex.quote(str(value))}")

    return " ; ".join(exports)
