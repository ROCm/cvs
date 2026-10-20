'''Shared install, version, log collection, and reporting helpers for AMD tools.'''

import os
import re
import tarfile
from collections import namedtuple
from datetime import datetime

from cvs.lib import globals
from cvs.lib.utils_lib import fail_test

log = globals.log
AMDTOOLS_PREFIX = "/opt/amdtools"
DOWNLOAD_PROGRESS_INTERVAL = 15
SHELL_UNSAFE_CHARS = ("'", '"', "`", "$", "\\", "\n")
VERSION_BODY = r"\d+\.\d+(?:\.\d+)*(?:-rc\.\d+)?"
VERSION_BODY_GROUPED = r"(\d+\.\d+(?:\.\d+)*)(?:-rc\.(\d+))?"
VERSION_END = r"(?!\.\d)(?![0-9A-Za-z])(?![-][rR][cC])"
_URL_VERSION_RE = re.compile(VERSION_BODY + VERSION_END)
_URL_VERSION_CANDIDATE_RE = re.compile(r"\d+\.\d+")
_VERSION_RE = re.compile(VERSION_BODY_GROUPED + VERSION_END)
PackageFlavour = namedtuple("PackageFlavour", ["pkg_type", "is_direct"])
NOT_AVAILABLE_TEXT = "not available on the remote system"


def node_label_from_file(cluster_dict, host):
    '''
    Build the artifact folder label for a node using ONLY the cluster file (no
    SSH). When the node has a usable vpc_ip that differs from the host key the
    label is "<ip>_<hostkey>"; when vpc_ip is missing/"NA" (or equals the host
    key) the label is just the host key so the name is not duplicated.

    This is the single source of truth for the node label so that the collected
    ANC log tree uses an identical folder name for the same node across runs. It
    deliberately does NOT resolve the hostname over SSH so the label stays stable
    and derivable from the cluster file alone.
    '''
    info = (cluster_dict.get("node_dict", {}) or {}).get(host, {}) or {}
    ip = info.get("vpc_ip")
    if not ip or ip == "NA":
        log.info("Node %s: vpc_ip not available in cluster file; using host name only for the node label", host)
        label = sanitize_path_component(f"{host}")
    elif ip == host:
        label = sanitize_path_component(f"{host}")
    else:
        label = sanitize_path_component(f"{ip}_{host}")
    log.info("Node %s: node == %s", host, label)
    return label


def as_bool(value, default=True):
    '''Interpret a config value (str/bool/None) as a boolean.'''
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def new_run_timestamp():
    '''Return a filesystem-safe timestamp (YYYYmmdd-HHMMSS) for a run folder.'''
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_run_log_folder(subpath, test_name, timestamp, node=None):
    '''
    Resolve this run's run_dir into an absolute tool log destination directory. The
    code appends the fixed subpath under run_dir, substituting "<node>"
    (the per-node "<ip>_<hostname>" label), "<test_name>", and "<timestamp>".

    run_dir comes from RunLayout (the same directory pytest writes its
    --html/--log-file into), so every tool artifact for a run lives together. The
    import is function-local to avoid the cvs.core <-> utils_lib import cycle
    (see cvs/lib/utils_lib.py).

    When ``node`` is None the "<node>" token is left intact (used for the
    run-start banner, which shows the pattern before any node is known).
    '''
    from cvs.core.run_layout import RunLayout

    prefix = str(RunLayout.get().run_dir)
    suffix = subpath.replace("<test_name>", test_name)
    suffix = suffix.replace("<timestamp>", timestamp)
    if node is not None:
        suffix = suffix.replace("<node>", node)
    return os.path.abspath(os.path.join(prefix, suffix))


def download_with_progress_snippet(url, dest, label, interval=DOWNLOAD_PROGRESS_INTERVAL):
    '''
    Build a shell snippet that downloads ``url`` to ``dest`` while emitting a
    periodic progress line, and exits non-zero if the download fails.

    ``wget -q`` is silent for the entire (potentially multi-minute) download,
    which would trip the exec's inactivity timeout on a slow link. Instead run
    wget in the background and, every ``interval`` seconds, echo the partial
    file size so the SSH channel keeps producing output while bytes flow. The
    whole thing is wrapped in a subshell whose exit status is wget's (via
    ``wait``), so it composes with the surrounding ``set -e && ...`` chain: a
    failed download aborts the install exactly as a plain ``wget && `` would.

    Because the heartbeat keeps the SSH channel active, the exec's inactivity
    timeout no longer catches a hung download. wget's own ``--timeout`` /
    ``--tries`` do that instead: a stalled read aborts after a few minutes,
    wget exits non-zero, and the ``set -e`` chain fails rather than hanging
    until the (large) install_timeout elapses.

    ``url`` and ``dest`` are single-quoted here; callers validate config values
    for embedded single quotes (see _assert_shell_safe) before interpolation.
    '''
    return (
        f"( wget -q --timeout=60 --tries=3 '{url}' -O '{dest}' & "
        f"wpid=$!; "
        f"while kill -0 $wpid 2>/dev/null; do "
        f"sleep {interval}; "
        f"if [ -f '{dest}' ]; then "
        f"echo \"{label} download progress: $(du -h '{dest}' 2>/dev/null | cut -f1) downloaded\"; "
        f"fi; "
        f"done; "
        f"wait $wpid )"
    )


def assert_shell_safe(anc_cfg, keys, tool):
    '''
    Reject config values that would break or subvert the remote shell strings.

    The install commands interpolate config values (anc_release_url,
    ANC_INSTALL_PATH, ...) into the remote command, both inside single quotes
    (which only a ``'`` breaks) and inside double quotes (where ``$``/backtick/
    ``\\`` still expand). A value containing any SHELL_UNSAFE_CHARS is rejected
    here so it can never reach the shell verbatim.
    '''
    for key in keys:
        value = anc_cfg.get(key)
        if value is None:
            continue
        text = str(value)
        bad = [c for c in SHELL_UNSAFE_CHARS if c in text]
        if bad:
            shown = ", ".join(repr(c) for c in bad)
            raise ValueError(
                f"{tool} config value {key}={value!r} contains unsafe shell "
                f"character(s) {shown}, which are not supported (they would break "
                f"or subvert the remote shell command). Remove them from the value."
            )


def validate_cluster_username(cluster_dict, tool):
    '''
    Return a problem string if the cluster file's ``username`` is unsafe to
    interpolate into remote shell commands, else None.

    The username comes from the cluster file (not the anc config), so it never
    passes through _assert_shell_safe, yet it is interpolated into remote
    commands (e.g. the ``sudo chown <user>`` in _pull_log_dir). A value with a
    space would split into two chown args; one with shell metacharacters could
    inject a command under sudo. Validate it at the fixture boundary. An
    empty/missing username is left to the connection layer to report.
    '''
    username = cluster_dict.get("username")
    if username is None or not str(username).strip():
        return None
    text = str(username)
    bad = [c for c in SHELL_UNSAFE_CHARS + (" ", "\t") if c in text]
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        return (
            f"cluster file username={username!r} contains unsafe character(s) {shown}; "
            f"{tool} interpolates the username into remote shell commands (e.g. chown), "
            f"so it must be a plain user name"
        )
    return None


def remote_user_tmp(user):
    '''
    Return a per-user remote temp DIRECTORY (``/tmp/<user>``) for ANC scratch
    files, so concurrent users running ANC on the same node do not collide on a
    shared, fixed-name path they cannot overwrite (another user's root/other
    ownership under /tmp/<file> would raise a permission error). Callers are
    responsible for ``mkdir -p`` before writing. The user is sanitised to a
    single safe path component.
    '''
    return f"/tmp/{sanitize_path_component(str(user))}"


def fail_unreachable_nodes(expected, out_dict, action, tool):
    '''
    fail_test every expected node absent from an exec result (unreachable).

    ``orch.exec`` only returns REACHABLE hosts, so a node that is down / did not
    respond simply drops out of ``out_dict``. Callers that judge pass/fail from
    ``out_dict`` alone would silently treat such a node as a success. This flags
    every expected host missing from ``out_dict`` as a failed ``action`` (e.g.
    "install", "ldconfig").

    Returns:
      list[str]: the expected hosts that were missing from ``out_dict``.
    '''
    missing = [host for host in expected if host not in (out_dict or {})]
    for host in missing:
        fail_test(f"{tool} {action} failed on {host}: node produced no output (unreachable / did not run)")
    return missing


def is_strict_version(version):
    '''
    True when ``version`` is ENTIRELY a valid ANC version token (no surrounding
    text, no trailing garbage): the stripped string fullmatches _VERSION_RE.

    Unlike parse_anc_version (which tolerates surrounding text for reading a
    version off node/URL output), this is the strict gate for a USER-configured
    value: ``1.7.0`` / ``1.7.0-rc.1`` pass; ``1.7.0-rc.bad`` / ``release-1.7.0``
    / ``1.7.0-RC.2`` / ``7`` do not. Returns False for blank/None.
    '''
    if not version:
        return False
    return _VERSION_RE.fullmatch(str(version).strip()) is not None


def parse_version(version):
    '''
    Parse an ANC version string into a comparable ``(base_tuple, rc)`` pair.

    ``base_tuple`` is the dotted-numeric release (e.g. ``(1, 7, 0)``); ``rc`` is
    the release-candidate number when the version carries a ``-rc.<n>`` suffix,
    else None (a final/base release). Returns None when no dotted-numeric version
    is present -- a bare integer does NOT qualify. Surrounding text is tolerated:
    the first ``X.Y[.Z...]`` token (with any attached ``-rc.<n>``) is read.

    Examples:
      "1.7.0"                -> ((1, 7, 0), None)
      "1.7.0-rc.1"           -> ((1, 7, 0), 1)
      "ANC 1.7.0-rc.1 build" -> ((1, 7, 0), 1)
      "1.4.9"                -> ((1, 4, 9), None)
      "7"                    -> None
    '''
    if not version:
        return None
    match = _VERSION_RE.search(str(version))
    if not match:
        return None
    base = tuple(int(part) for part in match.group(1).split("."))
    rc = int(match.group(2)) if match.group(2) is not None else None
    return (base, rc)


def compare_versions(left, right):
    '''
    Three-way compare two ANC version strings: -1 / 0 / 1 for left <, ==, >.

    Ordering rules (ANC-specific, intentionally NOT semver):
      - Base releases compare numerically, zero-padded to equal length
        (``1.6.0 < 1.7.0 < 1.8.0``).
      - A release candidate is treated as EQUAL to its own base release
        (``1.7.0-rc.1 == 1.7.0``) so an installed rc satisfies a request for the
        matching base and vice versa.
      - EXCEPT two candidates of the SAME base compare by rc number
        (``1.7.0-rc.1 < 1.7.0-rc.2 < 1.7.0-rc.10``).

    Raises ValueError if either string carries no parseable version, so callers
    can surface a clear config error rather than compare silently wrong.
    '''
    lv = parse_version(left)
    rv = parse_version(right)
    if lv is None or rv is None:
        raise ValueError(f"unparseable version: {left!r} vs {right!r}")
    (l_base, l_rc), (r_base, r_rc) = lv, rv

    width = max(len(l_base), len(r_base))
    l_padded = l_base + (0,) * (width - len(l_base))
    r_padded = r_base + (0,) * (width - len(r_base))
    if l_padded != r_padded:
        return -1 if l_padded < r_padded else 1

    # Same base: rc ranks equal to base, but rc-vs-rc compares by number.
    if l_rc is not None and r_rc is not None:
        if l_rc == r_rc:
            return 0
        return -1 if l_rc < r_rc else 1
    return 0


def version_satisfies(installed, required):
    '''
    True when an ``installed`` ANC version satisfies a ``required`` one, i.e.
    ``installed >= required`` under compare_versions. Used for the
    precheck-skip (per node) and post-install verification. Returns False on any
    unparseable input (treated as "cannot confirm it satisfies").
    '''
    try:
        return compare_versions(installed, required) >= 0
    except ValueError:
        return False


def detect_package_flavour(anc_release_url, tool):
    '''
    Infer the ANC package flavour AND packaging generation from the release name.

    Returns a PackageFlavour(pkg_type, is_direct):
      - pkg_type: one of ``"deb"``, ``"rpm"``, ``"tar"``.
      - is_direct: False for the LEGACY outer-tarball layout (name carries a
        ``-<flavour>-`` token, or a trailing ``-<flavour>.`` before the
        extension); True for the DIRECT 1.5.0+ layout (the URL is the artifact
        itself, flavour taken from the extension).

    The legacy token is checked FIRST because a legacy tar release also ends in
    ``.tar.gz`` -- extension alone cannot distinguish it from a direct tar.

    Raises:
      ValueError: if neither a legacy token nor a known direct extension is
      present in the name.
    '''
    name = anc_release_url.rsplit("/", 1)[-1].lower()
    # Legacy outer-tarball names carry the flavour as a `-<flavour>-` token, or a
    # trailing `-<flavour>.` before the extension (e.g. `...-1.4.9-deb.tar.gz`).
    # Match BOTH forms here, before the direct-extension fallback, so a legacy
    # `...-tar.tar.gz` is not mistaken for a DIRECT tar by its `.tar.gz` suffix.
    for pkg_type in ("deb", "rpm", "tar"):
        if f"-{pkg_type}-" in name or f"-{pkg_type}." in name:
            return PackageFlavour(pkg_type, False)
    if name.endswith(".deb"):
        return PackageFlavour("deb", True)
    if name.endswith(".rpm"):
        return PackageFlavour("rpm", True)
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return PackageFlavour("tar", True)
    raise ValueError(
        f"Cannot determine {tool} package flavour from archive name '{name}'; expected a legacy "
        f"'-deb-'/'-rpm-'/'-tar-' token or a direct '.deb'/'.rpm'/'.tar.gz' extension"
    )


def parse_version_from_url(anc_release_url):
    '''
    Extract the semantic version embedded in an ANC release URL, or None.

    Locates the FIRST dotted-numeric candidate in the filename (e.g. ``1.4.9``
    from ``...-1.4.9-deb-linux-x64.tar.gz``, ``1.5.5`` from
    ``...-1.5.5-x86_64.tar.gz`` and ``..._1.5.5_amd64.deb``) and requires the
    full ANC grammar to match AT that position. Returns None when the URL is
    empty, carries no candidate, or the first candidate is not a well-formed ANC
    version -- so a malformed leading token (``anc-1.7.0-rc.bad-glibc-2.31...``)
    is rejected rather than skipped in favour of the later ``2.31``.

    Used by the fail-fast config guard to abort (before contacting any node)
    when the user-supplied ``anc.anc_version`` is GREATER than the archive
    version (the archive could not satisfy the requested version).
    '''
    if not anc_release_url:
        return None
    name = anc_release_url.rsplit("/", 1)[-1]
    candidate = _URL_VERSION_CANDIDATE_RE.search(name)
    if not candidate:
        return None
    match = _URL_VERSION_RE.match(name, candidate.start())
    return match.group(0) if match else None


def sudo_prefix_snippet(prefix, label):
    '''
    Build a shell snippet that sets ``SUDO`` to "sudo" or "" for writing under
    ``prefix``, honouring the "auto: sudo only if not writable" policy.

    We cannot test the (not-yet-existing) prefix directly, so walk up to its
    deepest EXISTING ancestor and test whether THAT is writable by the current
    user. A user-writable ancestor (e.g. the home dir for a relocated install)
    means we can mkdir/extract without sudo, so files land user-owned and can be
    cleaned up later without root. A non-writable ancestor (e.g. /opt) selects
    sudo. The result is exported as ``$SUDO`` for the extract/rewrite commands.
    '''
    return (
        f"probe='{prefix}'; "
        f"while [ ! -e \"$probe\" ] && [ \"$probe\" != '/' ]; do probe=$(dirname \"$probe\"); done; "
        f"if [ -w \"$probe\" ]; then SUDO=''; else SUDO='sudo'; fi; "
        f"echo \"{label} install: prefix '{prefix}' -> $( [ -n \"$SUDO\" ] && echo 'sudo' || echo 'no-sudo (user-writable)' )\""
    )


def sanitize_path_component(value):
    '''Make a string safe to use as a single filesystem path component.'''
    return re.sub(r"[^\w.\-]", "_", value)


def chown_tree_to_runner(host, root_path):
    '''Recursively chown an extracted log tree to the current process's UID/GID.

    Best-effort: if the runner lacks privileges to chown (e.g. some entry is
    root-owned and the process is unprivileged), log a warning rather than
    failing the whole log-collection step.
    '''
    uid = os.getuid()
    gid = os.getgid()
    try:
        os.chown(root_path, uid, gid)
        for dirpath, dirnames, filenames in os.walk(root_path):
            for name in dirnames + filenames:
                try:
                    os.chown(os.path.join(dirpath, name), uid, gid)
                except OSError as exc:
                    log.warning("Node %s: could not chown %s: %s", host, os.path.join(dirpath, name), exc)
    except OSError as exc:
        log.warning("Node %s: could not chown log tree %s: %s", host, root_path, exc)


def is_within_directory(directory, target):
    '''
    True when ``target`` resolves to a path inside ``directory`` (inclusive).

    Uses ``realpath`` (not lexical ``abspath``) so SYMLINKS are resolved: a
    pre-existing ``directory/sub -> /outside`` symlink makes a regular
    ``sub/x`` member resolve outside and correctly fail the check. ``dest_dir``
    is created with ``exist_ok=True`` and may already contain such a link, so a
    lexical check would be bypassable. ``target``'s own leaf may not exist yet;
    ``realpath`` resolves the existing prefix and leaves the rest lexical, which
    is what we want (the parent chain is what a symlink attack subverts).
    '''
    real_dir = os.path.realpath(directory)
    real_target = os.path.realpath(target)
    return os.path.commonpath([real_dir, real_target]) == real_dir


def safe_tar_extract(tf, dest_dir):
    '''
    Extract ``tf`` into ``dest_dir`` rejecting any member that would escape it
    or create a special file.

    Used as the fallback on interpreters without ``extractall(filter="data")``
    (added in 3.12 / backported to 3.9.17+, 3.10.12+, 3.11.4+). Each member is
    validated, then extracted INDIVIDUALLY so no bulk ``extractall`` ever runs on
    unvalidated members. Only regular files and directories are allowed:
      - its final path (resolved through symlinks, see is_within_directory)
        must stay within ``dest_dir`` -- no absolute path, no ``..`` traversal,
        and no escape through a pre-existing symlink in the destination, and
      - symlink, hardlink, device, and FIFO members are rejected outright.
        Rejecting LINKS is deliberate: it stops the archive from planting a
        symlink that a LATER member then writes through (a chained-symlink
        sequence ``a -> .``; ``a/c -> ../out``; then ``a/c/pwn``). ANC log
        archives contain no links, so rejecting them costs nothing and closes
        the whole traversal class (device/FIFO rejection likewise prevents a
        node-controlled archive dropping a special file on the controller).
    Any violation raises ``tarfile.TarError`` (caught by the caller) instead of
    extracting.
    '''
    for member in tf.getmembers():
        if not (member.isfile() or member.isdir()):
            raise tarfile.TarError(f"unsafe non-regular member in tar archive ({member.type!r}): {member.name!r}")
        target = os.path.join(dest_dir, member.name)
        if not is_within_directory(dest_dir, target):
            raise tarfile.TarError(f"unsafe path in tar archive: {member.name!r}")
        tf.extract(member, dest_dir)


def pull_remote_dir(orch, host, user, log_dir, dest_dir, tool):
    '''
    Copy the entire tool log directory from the node into dest_dir.

    Tar the whole directory on the node, download the tarball, and extract it
    locally so the full tool log tree is preserved.

    Every orch call is scoped to the single ``host`` (``hosts=[host]``) so each
    node's log tree is pulled independently -- the per-node capture the suite
    relies on.

    Returns:
      tuple[str | None, str | None]: (extracted_dir, failure_reason).
      extracted_dir is the local path to the extracted directory on success;
      failure_reason is set (and path None) on any collection error.
    '''
    log_dir = log_dir.rstrip("/")
    base = os.path.basename(log_dir)
    parent = os.path.dirname(log_dir)
    # Per-user remote path so concurrent users on the same node do not collide on
    # a shared /tmp/anc_<base>.tar.gz owned by another user.
    user_tmp = remote_user_tmp(user)
    remote_tar = f"{user_tmp}/{tool.lower()}_{base}.tar.gz"

    # The log directory may be root-owned and unreadable by the SSH user. Build
    # the archive under sudo, then chown it back so SFTP can pull it. `&&` makes
    # a tar failure abort before exec tries to download a missing file.
    archive_cmd = (
        f"mkdir -p '{user_tmp}' && sudo tar -czf '{remote_tar}' -C '{parent}' '{base}' "
        f"&& sudo chown '{user}' '{remote_tar}'"
    )
    try:
        orch.exec(archive_cmd, hosts=[host], timeout=600)
    except Exception as exc:
        return None, f"could not archive log dir {log_dir}: {exc}"

    local_tar = os.path.join(dest_dir, f"{base}.tar.gz")
    # download_file suffixes the local path with '_<host>' (per-host collision
    # avoidance) and returns {host: actual_path}; use that, not local_tar.
    try:
        downloaded = orch.download_file(remote_tar, local_tar, hosts=[host])
    except Exception as exc:
        return None, f"could not download log archive from {log_dir}: {exc}"
    finally:
        try:
            orch.exec(f"rm -f '{remote_tar}'", hosts=[host], timeout=30)
        except Exception as exc:  # best-effort cleanup
            log.warning("Node %s: could not remove remote tar %s: %s", host, remote_tar, exc)

    local_tar = None
    if isinstance(downloaded, dict) and downloaded:
        # host-scoped download: prefer the host key, else the only entry.
        local_tar = downloaded.get(host) or next(iter(downloaded.values()))
    if not local_tar or not os.path.isfile(local_tar):
        return None, (
            f"could not archive log dir {log_dir}: remote archive was not created (check sudo/tar permissions on node)"
        )

    try:
        with tarfile.open(local_tar) as tf:
            # filter="data" sanitizes members (no absolute paths / "../" escapes)
            # and silences the 3.12+ extractall deprecation. The param was added
            # in 3.12 / backported to 3.9.17+, 3.10.12+, 3.11.4+; on older
            # interpreters (repo targets python>=3.9) fall back to the explicit
            # member-validating extractor, which enforces the same containment.
            try:
                tf.extractall(dest_dir, filter="data")
            except TypeError:
                safe_tar_extract(tf, dest_dir)
    except Exception as exc:
        return None, f"could not extract log archive for {log_dir}: {exc}"
    finally:
        try:
            os.remove(local_tar)
        except OSError:
            pass

    extracted_dir = os.path.join(dest_dir, base)
    # The archive may contain root-owned entries. Give
    # the whole extracted tree back to the user running the test so the logs are
    # readable/removable without sudo on this box.
    chown_tree_to_runner(host, extracted_dir)
    log.info("Node %s: %s logs copied to %s", host, tool, extracted_dir)

    return extracted_dir, None


def stash_report_link(request, rel_path, link_name, extras_attr):
    '''
    Stash a pytest-html url extra on the test node so it renders in the Links
    column. pytest-html 4.x reads links from report.extras; the repo-root
    pytest_runtest_makereport hook merges these stashed extras in for this test.
    Best-effort: a link problem never fails the test.
    '''
    if not rel_path:
        return
    try:
        import pytest_html

        extra = pytest_html.extras.url(rel_path, name=link_name)
        pending = getattr(request.node, extras_attr, [])
        pending.append(extra)
        setattr(request.node, extras_attr, pending)
    except Exception as exc:  # link is best-effort; file is already in zip
        log.warning(
            "%s '%s': could not add report link '%s': %s",
            extras_attr.split("_")[1].upper(),
            getattr(request.node, "name", "?"),
            link_name,
            exc,
        )


def replace_tops_and_extract_snippet(archive, prefix):
    '''Remove only archive-owned top-level folders before extracting.'''
    # Strip a leading ./ so the archive's top-level names match the install prefix.
    # Guard again before rm -rf in case a malformed archive yields a path-like name.
    return (
        f"echo 'Determining top-level tool folders...'; "
        f"tops=$( tar -tzf {archive} | sed 's#^\\./##' | awk -F/ '{{print $1}}' | grep -vE '^([.]{{1,2}})?$' | sort -u ); "
        f"$SUDO mkdir -p '{prefix}'; "
        f"echo 'Replacing top-level folders under {prefix}...'; "
        f"for name in $tops; do case \"$name\" in */*|.|..) continue;; esac; "
        f"echo \"  {prefix}/$name\"; $SUDO rm -rf '{prefix}'/\"$name\"; done; "
        f"echo 'Extracting archive to {prefix}...'; "
        f"$SUDO tar -xzf {archive} -C '{prefix}'; "
    )


def deck_status(reason):
    '''Return a Run Deck status from a failure reason.'''
    if reason is None:
        return "pass"
    if NOT_AVAILABLE_TEXT in reason:
        return "na"
    return "fail"


def attach_node_log_tarballs(request, mgr, results, test_name, timestamp, runner_base, tool, extras_attr):
    '''Archive collected logs per controller and add report links.'''
    present = [r for r in results if r.dest_dir and os.path.isdir(r.dest_dir) and os.listdir(r.dest_dir)]
    if not present:
        log.warning("%s '%s': no collected logs to attach to HTML report", tool, test_name)
        return {}
    os.makedirs(runner_base, exist_ok=True)
    hrefs = {}
    for r in present:
        node_dir = r.dest_dir.rstrip(os.sep)
        arcname = r.label or os.path.basename(node_dir)
        tar_path = os.path.join(runner_base, f"{r.label}_{test_name}_{timestamp}_{tool.lower()}_logs.tar.gz")
        try:
            with tarfile.open(tar_path, "w:gz") as tf:
                tf.add(node_dir, arcname=arcname)
        except Exception as exc:
            log.warning("%s '%s': could not archive logs for node %s: %s", tool, test_name, r.label, exc)
            continue
        link_name = f"{tool} logs: {r.label}" + (" (FAILED)" if r.reason else "")
        try:
            rel_path = mgr.add_html_to_report(
                tar_path, request=request, track_in_reports=True, test_name=test_name, passed=not r.reason
            )
            stash_report_link(request, rel_path, link_name, extras_attr)
            if rel_path:
                hrefs[r.label] = os.path.basename(rel_path)
        except Exception as exc:
            log.warning("%s '%s': could not attach logs for node %s: %s", tool, test_name, r.label, exc)
    return hrefs
