"""
env_lib.py

Utilities for safely constructing shell-compatible environment variable
exports, with controlled support for self-referential expansion (e.g. PATH).

Key guarantees:
- Prevents shell injection by quoting all user-provided values.
- Allows controlled expansion ONLY for `$KEY` where KEY is the variable being
  assigned (e.g., PATH=/x:$PATH, LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/y).
- Expansion is performed on the *remote shell*, not locally.

Also provides redact_secrets() for masking credential values in command
strings before they are logged.
"""

import re
import shlex

REDACTED = "<redacted>"

# Name words (split on "_", case-insensitive) that mark a secret: HF_TOKEN,
# SWITCH_SSH_PASSWORD, WANDB_API_KEY. Whole words only, so MAX_NUM_BATCHED_TOKENS
# and TOKENIZERS_PARALLELISM do not match.
_SECRET_WORDS = {"TOKEN", "PASSWORD", "PASSWD", "SECRET", "APIKEY"}
# A last word naming a location means the value is not the secret (HF_TOKEN_PATH, password_file).
_LOCATION_WORDS = {"FILE", "PATH", "DIR"}

_ASSIGNMENT = re.compile(r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z_][A-Za-z0-9_]*)=")

# One fragment of a shell word: shlex-quoted inside an outer single-quoted string
# ('"'"'a b'"'"'), ANSI-C quoted ($'a b'), single-quoted, double-quoted, or one bare
# character. Bare characters keep backslash escapes (a\ b) and exclude quotes,
# whitespace and shell operators so an outer quote is never consumed.
_FRAGMENT_PATTERN = (
    r"(?:'\"'\"'[^\n]*?'\"'\"'|\$'(?:[^'\\\n]|\\.)*'|'[^']*'|\"(?:[^\"\\]|\\.)*\"|\\.|[^\s'\";&|<>()\\])"
)
# A shell word is adjacent fragments: shlex.quote("a'b") is 'a'"'"'b', one word.
_VALUE_PATTERN = rf"(?:{_FRAGMENT_PATTERN})+"
_VALUE = re.compile(_VALUE_PATTERN)
# NAME=value as one quoted word (docker's -e 'HF_TOKEN=a b'): the value runs to the closing quote
# and on through any fragments joined to it.
_UNTIL_QUOTE = {"'": re.compile(r"[^'\n]*'"), '"': re.compile(r'[^"\n]*"')}

_SECRET_FLAG = re.compile(
    r"(?P<flag>(?<![A-Za-z0-9_-])--(?:(?:hf[-_])?token|password|api[-_]key))"
    rf"(?P<sep>=|(?:\s|\\\n)+){_VALUE_PATTERN}",
    re.IGNORECASE,
)


def _is_secret_name(name):
    words = name.upper().split("_")
    if words[-1] in _LOCATION_WORDS:
        return False
    if _SECRET_WORDS.intersection(words):
        return True
    return any(first == "API" and second == "KEY" for first, second in zip(words, words[1:]))


def _redact_assignments(text):
    parts = []
    pos = 0
    for match in _ASSIGNMENT.finditer(text):
        if match.start() < pos or not _is_secret_name(match["name"]):
            continue
        value_start = match.end()
        before = text[match.start() - 1] if match.start() else ""
        quoted = _UNTIL_QUOTE[before].match(text, value_start) if before in _UNTIL_QUOTE else None
        if quoted:
            value_end = quoted.end() - 1
            joined = _VALUE.match(text, quoted.end())
            if joined:
                # Keep the word's closing quote so the logged command stays balanced.
                value_end = joined.end() - (text[joined.end() - 1] == before)
        else:
            bare = _VALUE.match(text, value_start)
            value_end = bare.end() if bare else value_start
        if value_end == value_start:
            continue
        parts.append(text[pos:value_start])
        parts.append(REDACTED)
        pos = value_end
    if not parts:
        return text
    parts.append(text[pos:])
    return "".join(parts)


def redact_secrets(text):
    """
    Return text with credential values replaced by ``<redacted>``, for logging.

    Masks the value of:
      - NAME=value where NAME has TOKEN, PASSWORD, PASSWD, SECRET or API_KEY as a
        whole "_"-separated word and its last word is not FILE, PATH or DIR
        (case-insensitive);
      - the flags --token, --hf-token/--hf_token, --password and --api-key/--api_key.

    Use the result only for log output; always execute the original string.
    Non-string and empty input is returned unchanged. Runs in time linear in len(text).
    """
    if not text or not isinstance(text, str):
        return text
    if "=" in text:
        text = _redact_assignments(text)
    if "--" in text:
        text = _SECRET_FLAG.sub(lambda m: f"{m['flag']}{m['sep']}{REDACTED}", text)
    return text


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
