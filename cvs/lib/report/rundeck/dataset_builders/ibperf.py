'''
Copyright 2025 Advanced Micro Devices Inc.
All rights reserved.

Dataset builder for ibperf bandwidth results.
'''

from cvs.lib.report.rundeck.dataset_builders.registry import register_dataset_builder


def _sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _metric_samples(instance_results, metric):
    samples = []
    for instance, metrics in sorted((instance_results or {}).items(), key=lambda item: _sort_key(item[0])):
        if not isinstance(metrics, dict):
            continue
        try:
            samples.append((instance, float(metrics[metric])))
        except (KeyError, TypeError, ValueError):
            continue
    return samples


@register_dataset_builder("ibperf")
def build_ibperf_datasets(sources, _profile):
    results = sources.get("results") or sources.get("cvs_results_dict") or {}
    charts = {}
    table_rows = []

    for test_name, sizes in sorted(results.items()):
        if not isinstance(sizes, dict):
            continue
        points_by_qp = {}
        for msg_size, qp_results in sorted(sizes.items(), key=lambda item: _sort_key(item[0])):
            if not isinstance(qp_results, dict):
                continue
            for qp_count, node_results in sorted(qp_results.items(), key=lambda item: _sort_key(item[0])):
                if not isinstance(node_results, dict):
                    continue

                all_samples = []
                for node, instance_results in sorted(node_results.items()):
                    samples = _metric_samples(instance_results, "bw")
                    if not samples:
                        continue
                    all_samples.extend(value for _instance, value in samples)
                    min_instance, min_bw = min(samples, key=lambda sample: sample[1])
                    values = [value for _instance, value in samples]
                    table_rows.append(
                        [
                            test_name,
                            qp_count,
                            msg_size,
                            node,
                            len(values),
                            min_bw,
                            sum(values) / len(values),
                            max(values),
                            min_instance,
                        ]
                    )

                try:
                    size_number = int(msg_size)
                except (TypeError, ValueError):
                    continue
                if all_samples:
                    points_by_qp.setdefault(qp_count, []).append((size_number, sum(all_samples) / len(all_samples)))

        test_series = []
        for qp_count, points in sorted(points_by_qp.items(), key=lambda item: _sort_key(item[0])):
            test_series.append(
                {
                    "label": f"{test_name} · QP {qp_count}",
                    "points": sorted(points),
                }
            )
        if test_series:
            charts[test_name] = test_series

    return {
        "charts": {"bus_bw": charts},
        "results_table": {
            "headers": [
                "Test",
                "QP count",
                "Message size (bytes)",
                "Node",
                "NIC/GPU samples",
                "Min BW (Gb/s)",
                "Mean BW (Gb/s)",
                "Max BW (Gb/s)",
                "Minimum GPU/NIC",
            ],
            "rows": table_rows,
        },
    }
