'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re

from cvs.lib import globals
from cvs.lib.training.primus.primus_lib import PrimusTrainingJob
from cvs.lib.training.megatron.megatron_lib import MegatronTrainingJob
log = globals.log

def create_training_job(orch, variant_config, **kwargs):
    """Instantiate the correct TrainingJob class based on image and framework.

    Dispatch logic:
        1. If container image contains "primus" → PrimusTrainingJob, with the
           Primus backend derived from variant_config.framework by stripping
           the "_single" / "_distributed" topology suffix.
        2. Otherwise dispatch on the stripped framework name:
              megatron    → MegatronTrainingJob     (not yet implemented)
              jax         → JaxTrainingJob         (not yet implemented)
              torchtitan  → TorchTitanTrainingJob   (not yet implemented)

        Framework is always read from variant_config.framework — no extra
        config key or function argument is needed.

        Examples:
            image=primus,  framework="megatron_single"        → PrimusTrainingJob(framework="megatron")
            image=primus,  framework="torchtitan_distributed" → PrimusTrainingJob(framework="torchtitan")
            image=default, framework="megatron_distributed"   → MegatronTrainingJob
            image=default, framework="jax_single"             → JaxTrainingJob
            image=default, framework="torchtitan_distributed" → TorchTitanTrainingJob

    Args:
        orch:
            Orchestrator handle.
        variant_config:
            VariantConfig — framework field drives backend selection.
        **kwargs:
            Forwarded verbatim to the selected TrainingJob constructor.
            Expected keys: hf_token, micro_batch_size, global_batch_size,
            precision, result_dict, distributed_training, tune_model_params,
            scripts_dir, run_label.

    Returns:
        PrimusTrainingJob | MegatronTrainingJob | JaxTrainingJob | TorchTitanTrainingJob
    """
    image = orch.container_config.get("image", "")
    framework = re.sub(r'_(single|distributed)$', '', variant_config.framework)

    if re.search(r'primus', image, re.I):
        log.info(
            f"Image '{image}' matched Primus — "
            f"framework='{framework}' (from config framework='{variant_config.framework}') — "
            f"using PrimusTrainingJob"
        )
        return PrimusTrainingJob(
            orch, variant_config, primus_framework=framework, **kwargs
        )

    if framework == 'megatron':
        log.info(f"framework='{framework}' — MegatronTrainingJob not yet implemented")
        raise NotImplementedError("MegatronTrainingJob is not yet implemented")

    if framework == 'jax':
        log.info(f"framework='{framework}' — JaxTrainingJob not yet implemented")
        raise NotImplementedError("JaxTrainingJob is not yet implemented")

    if framework == 'torchtitan':
        log.info(f"framework='{framework}' — TorchTitanTrainingJob not yet implemented")
        raise NotImplementedError("TorchTitanTrainingJob is not yet implemented")

    raise ValueError(
        f"Unknown framework '{variant_config.framework}' — expected megatron, jax, or torchtitan "
        f"(with optional _single/_distributed suffix)"
    )
