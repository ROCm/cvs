'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Record ibperf results in the status_matrix shape the Run Deck renders.

Each NIC instance is compared with the fleet median of its run (test, message size, QP
count). The store holds ``groups`` (one per test, titled with the data direction) whose
cells are the QP counts (bandwidth) or latency percentiles, each carrying curves as
``series``. ``_meta`` carries the run facts, the ``weak_nics`` and ``results_table`` tables
and the ``nic_grid``
heatmap of the worst % of median per node x NIC at the largest size, which the builder
passes through so the profile can render them as their own sections. Raw per-instance
results live under ``_ibperf`` so every record call can rebuild the deck.
'''

from statistics import median

BW_OUTLIER_PCT = 95
BW_OUTLIER_MIN_SIZE = 64 << 10
LAT_OUTLIER_RATIO = 2.0
# Latency spans three decades on a linear axis; splitting at 32 KiB keeps small messages readable.
LATENCY_SPLIT = 32 << 10
OUTLIER_RULE = f"BW < {BW_OUTLIER_PCT}% of fleet median at ≥ 64 KiB; p99 > {LAT_OUTLIER_RATIO:g}× median p99"
# perftest's client initiates every test; read pulls data from the server.
_DIRECTION = {"read": "server → client", "lat": "client ↔ server"}


def size_text(size):
    for base, suffix in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if size >= base and size % base == 0:
            return f"{size // base} {suffix}"
    return f"{size} B"


def _tick(size):
    return size_text(size).replace(" KiB", "K").replace(" MiB", "M").replace(" GiB", "G").replace(" ", "")


def node_names(hosts):
    """{host: "node00", ...} in cluster order, the names the run card, groups and tables use."""
    return {str(host): f"node{index:02d}" for index, host in enumerate(hosts or [])}


def _instances(per_node, fields):
    """{host: {nic: {field: float}}} for instances whose first field parsed."""
    instances = {}
    for host, nics in (per_node or {}).items():
        for nic, metrics in (nics or {}).items():
            values = {}
            for field in fields:
                try:
                    values[field] = float((metrics or {})[field])
                except (KeyError, TypeError, ValueError):
                    continue
            if fields[0] in values:
                instances.setdefault(str(host), {})[str(nic)] = values
    return instances


def _cells(instances, field):
    return [(host, nic, m[field]) for host, nics in instances.items() for nic, m in nics.items() if field in m]


def _remember_variant(store, variant):
    meta = store.setdefault("_meta", {"suite": "ib_perf_bw_test"})
    if isinstance(variant, dict):
        for key in ("hosts", "expected_results", "ib_bw_test_list", "ib_lat_test_list"):
            if variant.get(key) is not None:
                meta[key] = variant[key]


def record_bw_results(results_store, bw_test, msg_size, qp_count, per_node, variant=None):
    """Record one bandwidth run (every node and NIC instance) and rebuild the deck."""
    instances = _instances(per_node, ("bw", "pps"))
    # A run whose instances all failed collection has nothing to plot.
    if not instances:
        return results_store
    _remember_variant(results_store, variant)
    raw = results_store.setdefault("_ibperf", {}).setdefault("bw", {})
    raw[f"{bw_test}|{qp_count}|{int(msg_size)}"] = instances
    return _rebuild(results_store)


def record_lat_results(results_store, lat_test, msg_size, per_node, variant=None):
    """Record one latency run (every node and NIC instance) and rebuild the deck."""
    instances = _instances(per_node, ("t_avg", "t_typical", "t_99_pct", "t_max"))
    if not instances:
        return results_store
    _remember_variant(results_store, variant)
    results_store.setdefault("_ibperf", {}).setdefault("lat", {})[f"{lat_test}|{int(msg_size)}"] = instances
    return _rebuild(results_store)


def _ordered(names, preferred):
    return sorted(names, key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))


def _qp_key(qp):
    return (int(qp) if str(qp).isdigit() else 0, str(qp))


def _short_test(test):
    return test.removeprefix("ib_").rsplit("_", 1)[0]


def _title(test, pairs):
    direction = _DIRECTION["lat"] if test.endswith("_lat") else _DIRECTION.get(_short_test(test), "client → server")
    return f"{test} · {direction}, {pairs} node pair{'' if pairs == 1 else 's'}"


def _threshold(expected, test, size, qp):
    try:
        return float(expected[test][str(size)][str(qp)])
    except (KeyError, TypeError, ValueError):
        return None


def _series(name, unit, points):
    return {"name": name, "unit": unit, "x_label": "message size", "points": points}


class _Deck:
    def __init__(self, store):
        self.meta = store.get("_meta") or {}
        raw = store.get("_ibperf") or {}
        self.bw = {}
        for key, instances in (raw.get("bw") or {}).items():
            test, qp, size = key.split("|")
            self.bw[(test, qp, int(size))] = instances
        self.lat = {}
        for key, instances in (raw.get("lat") or {}).items():
            test, size = key.split("|")
            self.lat[(test, int(size))] = instances
        hosts = list(self.meta.get("hosts") or [])
        for instances in list(self.bw.values()) + list(self.lat.values()):
            hosts.extend(host for host in instances if host not in hosts)
        self.names = node_names(hosts)
        # The suite pairs nodes in cluster order: an even index serves, the next one is its client.
        self.peer = {host: hosts[i ^ 1] for i, host in enumerate(hosts) if (i ^ 1) < len(hosts)}
        self.bw_tests = _ordered({t for t, _q, _s in self.bw}, list(self.meta.get("ib_bw_test_list") or []))
        self.lat_tests = _ordered({t for t, _s in self.lat}, list(self.meta.get("ib_lat_test_list") or []))
        self.qps = sorted({q for _t, q, _s in self.bw}, key=_qp_key)
        self.median = {run: median(v for _h, _n, v in _cells(inst, "bw")) for run, inst in self.bw.items()}
        self.findings, self.failed_runs = {}, set()
        self._find()

    def name(self, host):
        return self.names.get(host, host)

    def _flag(self, key, run, pct, value, med, limit, breach):
        found = self.findings.setdefault(key, {"tests": set(), "qps": set(), "sizes": set(), "breach": False})
        found["tests"].add(run[0])
        if run[1]:
            found["qps"].add(run[1])
        found["sizes"].add(run[2])
        found["breach"] = found["breach"] or breach
        worst = found.get("worst")
        if worst is None or (pct < worst[0] if key[0] == "BW" else pct > worst[0]):
            found["worst"] = (pct, value, med, limit)

    def _find(self):
        expected = self.meta.get("expected_results") or {}
        for run, instances in self.bw.items():
            test, qp, size = run
            limit, med = _threshold(expected, test, size, qp), self.median[run]
            for host, nic, value in _cells(instances, "bw"):
                pct = 100 * value / med if med else 100.0
                breach = limit is not None and value < limit
                if breach:
                    self.failed_runs.add((test, qp))
                if breach or (size >= BW_OUTLIER_MIN_SIZE and pct < BW_OUTLIER_PCT):
                    self._flag(("BW", host, nic), run, pct, value, med, limit, breach)
        for (test, size), instances in self.lat.items():
            p99 = _cells(instances, "t_99_pct")
            med = median(v for _h, _n, v in p99) if p99 else 0
            for host, nic, value in p99:
                if med and value / med > LAT_OUTLIER_RATIO:
                    self._flag(("Latency", host, nic), (test, None, size), value / med, value, med, None, False)

    def _bw_cell(self, test, qp):
        sizes = sorted(s for t, q, s in self.bw if (t, q) == (test, qp))
        median_points = [[_tick(s), round(self.median[(test, qp, s)], 2)] for s in sizes]
        slowest = [[_tick(s), min(v for _h, _n, v in _cells(self.bw[(test, qp, s)], "bw"))] for s in sizes]
        evaluated = any(_threshold(self.meta.get("expected_results") or {}, test, s, qp) is not None for s in sizes)
        status = "fail" if (test, qp) in self.failed_runs else ("pass" if evaluated else "na")
        return {
            "status": status,
            "series": [
                _series("Bandwidth, median NIC", "Gb/s", median_points),
                _series("Bandwidth, slowest NIC", "Gb/s", slowest),
            ],
        }

    def _lat_cells(self, test):
        sizes = sorted(s for t, s in self.lat if t == test)
        lines = {
            "median typical": ("t_typical", median),
            "median p99": ("t_99_pct", median),
            "worst NIC p99": ("t_99_pct", max),
        }
        cells = {}
        for label, (field, pick) in lines.items():
            points = []
            for size in sizes:
                values = [v for _h, _n, v in _cells(self.lat[(test, size)], field)]
                if values:
                    points.append((size, round(pick(values), 2)))
            small = [[_tick(s), v] for s, v in points if s <= LATENCY_SPLIT]
            large = [[_tick(s), v] for s, v in points if s >= LATENCY_SPLIT]
            outlier = label == "worst NIC p99" and any(
                k[0] == "Latency" and test in found["tests"] for k, found in self.findings.items()
            )
            cells[label] = {
                "status": "fail" if outlier else "na",
                "series": [
                    _series(f"Latency, up to {size_text(LATENCY_SPLIT)}", "µs", small),
                    _series(f"Latency, from {size_text(LATENCY_SPLIT)}", "µs", large),
                ],
            }
        return cells

    def heatmap(self):
        """Worst % of fleet median per node x NIC at the largest size over every bandwidth test and QP."""
        big = max(s for _t, _q, s in self.bw)
        runs = [(t, q, big) for t in self.bw_tests for q in self.qps if (t, q, big) in self.bw]
        hosts = sorted({h for run in runs for h in self.bw[run]}, key=self.name)
        nics = sorted({n for run in runs for nics in self.bw[run].values() for n in nics}, key=int)
        values = []
        for host in hosts:
            row = []
            for nic in nics:
                pcts = [
                    100 * self.bw[run][host][nic]["bw"] / self.median[run]
                    for run in runs
                    if nic in self.bw[run].get(host, {})
                ]
                row.append(round(min(pcts), 1) if pcts else None)
            values.append(row)
        return {
            "name": f"NIC bandwidth at {size_text(big)}, worst % of fleet median over every test and QP",
            "unit": f"% of the median NIC for the same test and QP · red below {BW_OUTLIER_PCT}%",
            "rows": [f"{self.name(h)} ↔ {self.name(self.peer[h])}" if h in self.peer else self.name(h) for h in hosts],
            "cols": [f"NIC {nic}" for nic in nics],
            "values": values,
            "threshold": BW_OUTLIER_PCT,
            "direction": "higher",
        }

    def groups(self):
        pairs = len(self.names) // 2
        groups = {}
        for test in self.bw_tests:
            nodes = {
                f"QP {qp}": self._bw_cell(test, qp)
                for qp in self.qps
                if any((t, q) == (test, qp) for t, q, _s in self.bw)
            }
            groups[_title(test, pairs)] = {"nodes": nodes}
        for test in self.lat_tests:
            groups[_title(test, pairs)] = {"nodes": self._lat_cells(test)}
        return groups

    def results_table(self):
        """One row per bandwidth test x QP x message size, then per latency test x message size."""
        expected = self.meta.get("expected_results") or {}
        rows = []
        for test in self.bw_tests:
            for qp in self.qps:
                for size in sorted(s for t, q, s in self.bw if (t, q) == (test, qp)):
                    cells = _cells(self.bw[(test, qp, size)], "bw")
                    low, host, nic = min((v, h, n) for h, n, v in cells)
                    limit = _threshold(expected, test, size, qp)
                    status = "—" if limit is None else ("fail" if low < limit else "pass")
                    median_bw = round(self.median[(test, qp, size)], 2)
                    slowest = f"{self.name(host)} NIC {nic}"
                    limit_text = "—" if limit is None else f"{limit:g}"
                    rows.append(
                        [test, qp, _tick(size), median_bw, low, slowest, "—", "—", "—", limit_text, status, len(cells)]
                    )
        for test in self.lat_tests:
            for size in sorted(s for t, s in self.lat if t == test):
                instances = self.lat[(test, size)]
                typical = [v for _h, _n, v in _cells(instances, "t_typical")]
                p99 = [v for _h, _n, v in _cells(instances, "t_99_pct")]
                rows.append(
                    [test, "—", _tick(size), "—", "—", "—"]
                    + [round(median(v), 2) if v else "—" for v in (typical, p99)]
                    + [max(p99) if p99 else "—", "—", "—", len(_cells(instances, "t_avg"))]
                )
        headers = [
            "Test",
            "QP",
            "Message size",
            "Median BW",
            "Min BW",
            "Slowest NIC",
            "Median typical",
            "Median p99",
            "Worst p99",
            "BW threshold",
            "Status",
            "NICs",
        ]
        units = [None, None, None, "Gbit/s", "Gbit/s", None, "µs", "µs", "µs", "Gbit/s", None, None]
        align = [
            "left",
            "right",
            "right",
            "right",
            "right",
            "left",
            "right",
            "right",
            "right",
            "right",
            "left",
            "right",
        ]
        return {"headers": headers, "units": units, "align": align, "rows": rows}

    def weak_nics(self):
        def order(item):
            (kind, host, nic), found = item
            return (kind != "BW", found["worst"][0] if kind == "BW" else -found["worst"][0], self.name(host))

        rows = []
        for (kind, host, nic), found in sorted(self.findings.items(), key=order):
            pct, value, med, limit = found["worst"]
            sizes = sorted(found["sizes"])
            span = size_text(sizes[0]) if len(sizes) == 1 else f"{size_text(sizes[0])} – {size_text(sizes[-1])}"
            if kind == "BW":
                relative, unit = f"{pct:.0f}% of median", "Gb/s"
            else:
                relative, unit = f"{pct:.1f}× median", "µs p99"
            limit_text = f" (threshold {limit:g})" if limit is not None else ""
            rows.append(
                [
                    "below threshold" if found["breach"] else "outlier",
                    kind,
                    f"{self.name(host)} · NIC {nic}",
                    self.name(self.peer[host]) if host in self.peer else "—",
                    ", ".join(_short_test(t) for t in _ordered(found["tests"], self.bw_tests + self.lat_tests)),
                    ", ".join(sorted(found["qps"], key=_qp_key)) or "—",
                    f"{span} ({len(sizes)} sizes)" if len(sizes) > 1 else span,
                    relative,
                    f"{value:.1f} vs {med:.1f} {unit}{limit_text}",
                ]
            )
        if not rows:
            rows.append(["none", "—", "no NIC below a BW threshold or the outlier rule", "—", "—", "—", "—", "—", "—"])
        headers = ["Status", "Kind", "Node · NIC", "Pair peer", "Tests", "QPs", "Sizes", "Relative", "Worst vs median"]
        return {"headers": headers, "rows": rows}


def _rebuild(store):
    deck = _Deck(store)
    store["groups"] = deck.groups()
    store["_meta"]["weak_nics"] = deck.weak_nics()
    store["_meta"]["results_table"] = deck.results_table()
    store["_meta"]["nic_grid"] = {"heatmaps": [deck.heatmap()]} if deck.bw else {}
    return store


__all__ = ["OUTLIER_RULE", "node_names", "record_bw_results", "record_lat_results", "size_text"]
