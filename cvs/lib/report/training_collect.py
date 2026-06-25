'''Collect training results into ``training_res_dict`` for suite reports.'''

from __future__ import annotations

from typing import MutableMapping

from cvs.lib.report.training_normalize import megatron_to_training_res_dict


def record_megatron_training_results(
    training_res_dict: MutableMapping[str, dict],
    mt_obj,
) -> dict[str, dict]:
    """Normalize ``MegatronLlamaTrainingJob.training_results_dict`` into the session store."""
    nodes = megatron_to_training_res_dict(
        getattr(mt_obj, "training_results_dict", {}) or {},
        getattr(mt_obj, "host_list", []) or [],
    )
    training_res_dict.clear()
    training_res_dict.update(nodes)
    return nodes
