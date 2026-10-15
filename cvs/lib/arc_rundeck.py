'''Structured ARC summary records for the status matrix Run Deck.'''


def make_meta(cluster_dict, config_dict, suite_name, timestamp):
    '''Assemble Run Deck metadata.'''
    arc = config_dict.get("arc", {}) if isinstance(config_dict, dict) else {}
    return {
        "cluster": (cluster_dict or {}).get("cluster_name") or (cluster_dict or {}).get("name") or "—",
        "version": arc.get("arc_version") or "—",
        "version_label": "ARC version",
        "suite": suite_name or "arc",
        "generated_at": timestamp or "",
    }


def items_summary_line(summary):
    '''Format ARC's selected-item counts for a matrix cell.'''
    if not isinstance(summary, dict):
        return ""
    overall = summary.get("overall_summary")
    if not isinstance(overall, dict):
        return ""
    keys = ("total_selected_items", "total_passed", "total_completed", "total_failed", "total_skipped", "total_not_run")
    values = [overall.get(key, summary.get(key, 0)) for key in keys]
    return (
        f"Items: {values[0]} Total | {values[1]} PASSED, {values[2]} COMPLETED, "
        f"{values[3]} FAILED, {values[4]} SKIPPED, {values[5]} NOT RUN"
    )


def failed_items(summary):
    '''Return only actual failed or cancelled ARC items.'''
    if not isinstance(summary, dict):
        return []
    results = summary.get("results")
    if not isinstance(results, dict):
        return []
    items = []
    for key, rec in results.items():
        if not isinstance(rec, dict) or str(rec.get("status", "")).upper() not in ("FAILED", "CANCELLED"):
            continue
        message = ": ".join(str(part) for part in (rec.get("rc"), rec.get("summary")) if part)
        items.append({"name": rec.get("name") or key, "status": "fail", "message": message})
    return items


def build_node_record(*, status, summary, errors_json_href, log_tarball_href):
    '''Build one controller's status matrix record.'''
    return {
        "status": status,
        "items_summary": items_summary_line(summary),
        "items": failed_items(summary) if status == "fail" else [],
        "errors_json_href": errors_json_href or "",
        "log_tarball_href": log_tarball_href or "",
    }
