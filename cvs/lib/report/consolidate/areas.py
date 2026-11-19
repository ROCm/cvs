'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Areas that group suites in the consolidated report, and the test files behind older decks.

A deck that records its own ``area`` keeps it. Decks written before that field
existed are looked up by ``suite_id`` in ``AREA_BY_SUITE_ID``, which covers every
suite that wrote a deck up to release 0.3.0. The table never grows: every newer
deck writes its own area. The same goes for ``older_deck_stems``, because every
newer deck also records the test file that wrote it.
'''

AREAS = ("Inference", "Training", "Communication", "Health", "AMD Node Check")
UNCATEGORIZED = "Uncategorized"

_ANC_TESTS = ("babel", "basic", "computerocker", "cpu", "gemm", "gpu", "memrocker", "oblex", "pcie", "ualink", "xgmi")
_ARC_TESTS = ("anc", "diag", "disruptive", "disruptive_flows", "examples", "flows", "telemetry", "workload")

SUITE_IDS_BY_AREA = {
    "Inference": ("atom", "sglang", "vllm", "xdit"),
    "Training": ("aorta", "jaxmaxtext", "megatron", "torchtitan"),
    "Communication": ("rccl",),
    "Health": (
        "agfhc_cvs",
        "csp_qual_agfhc",
        "rvs_cvs",
        "transferbench_cvs",
        "host_configs_cvs",
        "preflight_checks",
        "ib_perf_bw_test",
        "mori_benchmark_test",
        *("arc_test_" + name for name in _ARC_TESTS),
    ),
    "AMD Node Check": tuple("anc_test_" + name for name in _ANC_TESTS),
}

AREA_BY_SUITE_ID = {suite_id: area for area, suite_ids in SUITE_IDS_BY_AREA.items() for suite_id in suite_ids}

_STEMS_BY_SUITE_ID = {
    "rccl": ("rccl_perf", "rccl_regression", "rccl_pairwise"),
    "aorta": ("aorta_single", "aorta_distributed"),
    "jaxmaxtext": ("jaxmaxtext_single", "jaxmaxtext_distributed"),
    "megatron": ("megatron_single", "megatron_distributed"),
    "torchtitan": ("torchtitan_single", "torchtitan_distributed"),
    "sglang": ("sglang_single", "sglang_distributed", "sglang_disagg_distributed"),
    "vllm": ("vllm_single", "vllm_distributed"),
    "xdit": (
        "xdit_flux_dev_single",
        "xdit_flux_dev_distributed",
        "xdit_wan22_14b_single",
        "xdit_wan22_14b_diffusers_single",
        "xdit_wan22_14b_diffusers_distributed",
        "xdit_wan22_i2v_14b_diffusers_single",
        "xdit_wan22_i2v_14b_diffusers_distributed",
        "xdit_wan21_t2v_14b_diffusers_single",
        "xdit_wan21_t2v_14b_diffusers_distributed",
    ),
}


def older_deck_stems(suite_id):
    '''The test-file stems that wrote decks of a known ``suite_id`` before decks recorded their stem, or None.

    A suite_id not listed in ``_STEMS_BY_SUITE_ID`` came only from the test file of the same name.
    '''
    if suite_id not in AREA_BY_SUITE_ID:
        return None
    return _STEMS_BY_SUITE_ID.get(suite_id, (suite_id,))


def suite_id_for_stem(stem):
    '''Return the known suite_id a test-file stem belongs to, e.g. ``rccl`` for ``rccl_perf``, or None.'''
    if not stem:
        return None
    if stem in AREA_BY_SUITE_ID:
        return stem
    prefixes = [suite_id for suite_id in AREA_BY_SUITE_ID if stem.startswith(suite_id + "_")]
    return max(prefixes, key=len) if prefixes else None


def resolve_area(deck, stem=None):
    '''Return the area of a deck, or None when it is unknown.

    ``deck`` is the parsed deck JSON, or None when the file could not be read; the
    test-file ``stem`` is used only in that case.
    '''
    if isinstance(deck, dict):
        declared = deck.get("area")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
        return AREA_BY_SUITE_ID.get(deck.get("suite_id"))
    return AREA_BY_SUITE_ID.get(suite_id_for_stem(stem))


def ordered_areas(used):
    '''The report's areas in display order: the standard ones, then any other declared area, then Uncategorized.'''
    used = set(used)
    extra = sorted(area for area in used if area not in AREAS and area != UNCATEGORIZED)
    tail = [UNCATEGORIZED] if UNCATEGORIZED in used else []
    return list(AREAS) + extra + tail
