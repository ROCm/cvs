'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Record ibperf results in the status_matrix shape the Run Deck renders. Each NIC instance is
compared with the fleet median of its run. Raw per-instance results live under ``_ibperf`` so
every record call rebuilds the deck.
'''

from statistics import median

BW_OUTLIER_PCT = 95
LAT_OUTLIER_RATIO = 2.0
# Latency spans three decades on a linear axis; splitting at 32 KiB keeps small messages readable.
LATENCY_SPLIT = 32 << 10
OUTLIER_RULE = f"BW < {BW_OUTLIER_PCT}% of fleet median at the largest size; p99 > {LAT_OUTLIER_RATIO:g}× median p99"
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
    for host, nics in per_node.items():
        for nic, metrics in nics.items():
            values = {}
            for field in fields:
                try:
                    values[field] = float(metrics[field])
                except (KeyError, TypeError, ValueError):
                    continue
            if fields[0] in values:
                instances.setdefault(str(host), {})[str(nic)] = values
    return instances


def _cells(instances, field):
    return [(host, nic, m[field]) for host, nics in instances.items() for nic, m in nics.items() if field in m]


def _remember_variant(store, variant):
    meta = store.setdefault("_meta", {})
    for key in ("hosts", "expected_results", "ib_bw_test_list", "ib_lat_test_list"):
        if variant and variant.get(key) is not None:
            meta[key] = variant[key]


def record_bw_results(results_store, bw_test, msg_size, qp_count, per_node, variant=None):
    """Record one bandwidth run (every node and NIC instance) and rebuild the deck."""
    instances = _instances(per_node, ("bw",))
    # A run whose instances all failed collection has nothing to plot.
    if not instances:
        return results_store
    _remember_variant(results_store, variant)
    results_store.setdefault("_ibperf", {}).setdefault("bw", {})[f"{bw_test}|{qp_count}|{int(msg_size)}"] = instances
    return _rebuild(results_store)


def record_lat_results(results_store, lat_test, msg_size, per_node, variant=None):
    """Record one latency run (every node and NIC instance) and rebuild the deck."""
    instances = _instances(per_node, ("t_avg", "t_typical", "t_99_pct"))
    if not instances:
        return results_store
    _remember_variant(results_store, variant)
    results_store.setdefault("_ibperf", {}).setdefault("lat", {})[f"{lat_test}|{int(msg_size)}"] = instances
    return _rebuild(results_store)


def _ordered(names, preferred):
    return sorted(names, key=lambda name: (preferred.index(name) if name in preferred else len(preferred), name))


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
        self.expected = self.meta.get("expected_results") or {}
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
        self.qps = sorted({q for _t, q, _s in self.bw}, key=int)
        self.median = {run: median(v for _h, _n, v in _cells(inst, "bw")) for run, inst in self.bw.items()}
        self._find()

    def name(self, host):
        return self.names.get(host, host)

    def _breach(self, run):
        test, qp, size = run
        limit = _threshold(self.expected, test, size, qp)
        return limit is not None and any(v < limit for _h, _n, v in _cells(self.bw[run], "bw"))

    def _find(self):
        self.failed = {(t, q) for (t, q, s) in self.bw if self._breach((t, q, s))}
        self.lat_outliers = set()
        for (test, _size), instances in self.lat.items():
            p99 = [v for _h, _n, v in _cells(instances, "t_99_pct")]
            med = median(p99) if p99 else 0
            if med and max(p99) / med > LAT_OUTLIER_RATIO:
                self.lat_outliers.add(test)

    def _bw_cell(self, test, qp):
        runs = [(test, qp, s) for s in sorted(s for t, q, s in self.bw if (t, q) == (test, qp))]
        median_points = [[_tick(run[2]), round(self.median[run], 2)] for run in runs]
        slowest = [[_tick(run[2]), min(v for _h, _n, v in _cells(self.bw[run], "bw"))] for run in runs]
        evaluated = any(_threshold(self.expected, test, run[2], qp) is not None for run in runs)
        status = "fail" if (test, qp) in self.failed else ("pass" if evaluated else "na")
        return {
            "status": status,
            "series": [
                _series("Bandwidth, median NIC", "Gbit/s", median_points),
                _series("Bandwidth, slowest NIC", "Gbit/s", slowest),
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
            outlier = label == "worst NIC p99" and test in self.lat_outliers
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
        rows = []
        for test in self.bw_tests:
            for qp in self.qps:
                for size in sorted(s for t, q, s in self.bw if (t, q) == (test, qp)):
                    run = (test, qp, size)
                    cells = _cells(self.bw[run], "bw")
                    low, host, nic = min((v, h, n) for h, n, v in cells)
                    median_bw, slowest = round(self.median[run], 2), f"{self.name(host)} NIC {nic}"
                    limit = _threshold(self.expected, test, size, qp)
                    status = "fail" if self._breach(run) else ("pass" if limit is not None else "—")
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
        align = ["left"] + ["right"] * 4 + ["left"] + ["right"] * 4 + ["left", "right"]
        return {"headers": headers, "units": units, "align": align, "rows": rows}


def _rebuild(store):
    deck = _Deck(store)
    store["groups"] = deck.groups()
    store["_meta"]["results_table"] = deck.results_table()
    store["_meta"]["nic_grid"] = {"heatmaps": [deck.heatmap()]} if deck.bw else {}
    return store


__all__ = ["OUTLIER_RULE", "node_names", "record_bw_results", "record_lat_results", "size_text"]
