#!/usr/bin/env python3
#
# Copyright 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Security scanner for the CVS git hooks. Flags internal-infrastructure leaks in
# added/changed lines: internal *.amd.com endpoints, real cluster hostnames
# (helios racks), non-standard https ports on routable hosts, and committed
# secrets/tokens. Public AMD docs domains and the short helios *family* names
# stay allowed via explicit allowlists, so legitimate references do not trip.
#
# Pure stdlib, no type annotations (CVS convention). The scan_* functions take
# plain strings so they are unit-testable without git. The CLI reads a unified
# diff (from `--staged`, `--range A..B`, or stdin) and scans only added lines.
#
# Suppress a single line that is a false positive by adding the pragma token
# `cvs-security-allow` somewhere on that line.

import re
import subprocess
import sys

PRAGMA = "cvs-security-allow"

# Public, intentionally-published AMD hosts. Any other *.amd.com host in new
# content is treated as an internal endpoint and blocked. Exact-host match: a
# new public domain must be added here deliberately.
PUBLIC_AMD_HOSTS = frozenset(
    {
        "amd.com",
        "www.amd.com",
        "rocm.docs.amd.com",
        "instinct.docs.amd.com",
        "rocm.blogs.amd.com",
        "repo.amd.com",
    }
)

# A host token that ends in amd.com (optionally with sub-labels), scheme-independent
# so bare `host.amd.com:8443` references are caught too.
AMD_HOST_RE = re.compile(r"(?<![A-Za-z0-9._-])((?:[A-Za-z0-9-]+\.)*amd\.com)(?![A-Za-z0-9-])", re.IGNORECASE)

# https://host:port  -- capture host and explicit port.
HTTPS_PORT_RE = re.compile(r"https://([A-Za-z0-9._-]+):(\d+)", re.IGNORECASE)

# Real helios cluster hostnames. The short family names (heliosr, heliosp,
# heliosm, heliosr-rck, heliosm-rck) and underscore ARC names (heliosr_mfg_l11)
# are allowed; a name is a real host only once it carries datacenter
# coordinates: a building code like 1b114/2b805 right after the family, or
# `-rck-` followed by further rack coordinates. Hyphen-joined only, so the
# underscore ARC names never match.
REAL_HELIOS_HOST_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:ct)?helios[rmp]"
    r"(?:"
    r"-\d+b\d+(?:-[0-9a-z]+)+"  # family-1b114-g07-1
    r"|"
    r"-rck(?:-[0-9a-z]+)+"  # family-rck-g02-j17-7
    r")"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Loopback / private / link-local / placeholder hosts never count as a leaked
# listening endpoint.
_PRIVATE_HOST_RE = re.compile(
    r"^(?:localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|"
    r"10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|"
    r"169\.254\.\d+\.\d+)$",
    re.IGNORECASE,
)

# Values that are obviously not real secrets.
_PLACEHOLDER_HINTS = (
    "changeme",
    "example",
    "placeholder",
    "redacted",
    "your_",
    "your-",
    "dummy",
    "sample",
    "<",
    ">",
    "${",
    "os.environ",
    "getenv",
    "xxxx",
    "****",
)

# Assembled at runtime so the verbatim PEM header never appears in this file;
# GitHub push-protection flags the contiguous literal even inside a scanner's
# own detection pattern.
_PEM_KEY_TAIL = "PRIVATE" + " KEY-----"

# High-confidence secret token shapes (the value itself identifies the vendor).
_SECRET_TOKEN_RES = (
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?" + _PEM_KEY_TAIL)),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("huggingface-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("private-key-file", re.compile(r"-----BEGIN OPENSSH " + _PEM_KEY_TAIL)),
)

# Generic `secret = "value"` assignment; the value is vetted separately so
# placeholders and env lookups do not trip it.
_GENERIC_SECRET_RE = re.compile(
    r"(?i)\b(pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|auth[_-]?token)\b"
    r"\s*[:=]\s*"
    r"""(['"])([^'"]{8,})(\2)"""
)


def _looks_like_placeholder(value):
    low = value.lower()
    if any(hint in low for hint in _PLACEHOLDER_HINTS):
        return True
    # all one repeated character, e.g. "********"
    if len(set(value)) <= 1:
        return True
    return False


def _looks_like_secret_value(value):
    if _looks_like_placeholder(value):
        return False
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    # Long opaque string, or a mixed alnum string of reasonable length.
    if len(value) >= 20 and has_alpha:
        return True
    if len(value) >= 10 and has_digit and has_alpha:
        return True
    return False


def scan_line(text):
    """Return a list of (rule, message) findings for one line of content.

    Pure function over a string -- no git, no filesystem. A line carrying the
    `cvs-security-allow` pragma yields no findings.
    """
    if PRAGMA in text:
        return []

    findings = []

    for host in AMD_HOST_RE.findall(text):
        if host.lower() not in PUBLIC_AMD_HOSTS:
            findings.append(
                (
                    "internal-amd-host",
                    "internal AMD host '{}' (only public AMD docs domains are allowed)".format(host),
                )
            )

    for host, port in HTTPS_PORT_RE.findall(text):
        if port == "443":
            continue
        if _PRIVATE_HOST_RE.match(host) or "<" in host or ">" in host:
            continue
        findings.append(
            (
                "exposed-https-port",
                "https endpoint '{}:{}' exposes a non-standard listening port".format(host, port),
            )
        )

    for match in REAL_HELIOS_HOST_RE.finditer(text):
        findings.append(
            (
                "real-cluster-hostname",
                "real cluster hostname '{}' (use a generic name like nodeA / heliosr)".format(match.group(0)),
            )
        )

    for rule, pattern in _SECRET_TOKEN_RES:
        if pattern.search(text):
            findings.append((rule, "possible {} committed".format(rule.replace("-", " "))))

    gm = _GENERIC_SECRET_RE.search(text)
    if gm and _looks_like_secret_value(gm.group(3)):
        findings.append(
            (
                "hardcoded-secret",
                "hardcoded secret in '{}=...' assignment".format(gm.group(1)),
            )
        )

    return findings


# Files whose job is to contain these patterns (the scanner and its tests).
_EXEMPT_BASENAMES = frozenset({"security_scan.py", "test_security_scan.py"})


def _is_exempt_path(path):
    if path is None:
        return False
    base = path.rsplit("/", 1)[-1]
    return base in _EXEMPT_BASENAMES


def iter_added_lines(diff_text):
    """Yield (path, new_lineno, content) for every added line in a unified diff."""
    current = None
    new_lineno = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current = None if path == "/dev/null" else path
            continue
        if raw.startswith("--- "):
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            new_lineno = int(m.group(1)) if m else 0
            continue
        if raw.startswith("+"):
            yield current, new_lineno, raw[1:]
            new_lineno += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith(" "):
            new_lineno += 1


def scan_diff(diff_text):
    """Scan a unified diff; return a list of (path, lineno, rule, message)."""
    results = []
    for path, lineno, content in iter_added_lines(diff_text):
        if _is_exempt_path(path):
            continue
        for rule, message in scan_line(content):
            results.append((path, lineno, rule, message))
    return results


def _git_diff(args):
    return subprocess.run(
        ["git", "diff", "--unified=0"] + args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _read_diff(argv):
    if "--stdin" in argv:
        return sys.stdin.read()
    if "--staged" in argv or "--cached" in argv:
        return _git_diff(["--cached"])
    for i, a in enumerate(argv):
        if a == "--range" and i + 1 < len(argv):
            return _git_diff([argv[i + 1]])
    return _git_diff(["--cached"])


def main(argv):
    diff_text = _read_diff(argv)
    findings = scan_diff(diff_text)
    if not findings:
        return 0

    sys.stderr.write("\nsecurity-scan: blocked -- internal-infrastructure / secret leaks in new lines:\n\n")
    for path, lineno, rule, message in findings:
        where = "{}:{}".format(path or "<unknown>", lineno)
        sys.stderr.write("  {}  [{}] {}\n".format(where, rule, message))
    sys.stderr.write(
        "\nsecurity-scan: scrub the values above (or append the '" + PRAGMA + "' pragma to a line\n"
        "that is a verified false positive), then retry. Bypass the hook with --no-verify.\n\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
