'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.lib import globals

log = globals.log


class _LegacyVllmInferenceJobPlaceholder:
    """``InferenceJobFactory`` no longer constructs a host+docker vLLM ``InferenceBaseJob``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InferenceJobFactory no longer builds a host+docker vLLM InferenceBaseJob. "
            "Use ``cvs.lib.inference.vllm_single.VllmJob`` with ``ContainerOrchestrator`` "
            "from the tests under ``cvs.tests.inference.vllm``."
        )


class _LegacyInferenceXAtomInferenceJobPlaceholder:
    """``InferenceJobFactory`` no longer constructs a host+docker InferenceX ATOM ``InferenceBaseJob``."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "InferenceMax is deprecated. Use ``cvs.lib.inference.inferencex_atom_orch.InferenceXAtomJob`` "
            "with ``ContainerOrchestrator`` from the tests under ``cvs.tests.inference.inferencex_atom`` "
            "(``inferencex_atom_single`` suite and schema_version 1 configs)."
        )


class InferenceJobFactory:
    """Factory class to create inference job instances based on framework type."""

    # Registry of supported frameworks
    _FRAMEWORK_CLASSES = {
        'vllm': _LegacyVllmInferenceJobPlaceholder,
        'inferencemax': _LegacyInferenceXAtomInferenceJobPlaceholder,
        'inferencex_atom': _LegacyInferenceXAtomInferenceJobPlaceholder,
    }

    @classmethod
    def _detect_framework(cls, inference_config_dict):
        """
        Auto-detect framework type from inference config dictionary.

        Args:
            inference_config_dict: Infrastructure configuration dictionary

        Returns:
            Detected framework name ('vllm' or 'inferencex_atom')

        Detection logic:
            - If 'inferencemax_repo' is present → inferencex_atom (InferenceMax deprecated)
            - If 'vllm_script_path' is present → vLLM
            - Otherwise → vLLM (default)
        """
        if 'inferencemax_repo' in inference_config_dict:
            log.warning(
                "inferencemax_repo detected; InferenceMax is deprecated — "
                "use the inferencex_atom_single suite instead"
            )
            return 'inferencex_atom'
        elif 'vllm_script_path' in inference_config_dict:
            return 'vllm'
        else:
            # Default to vLLM if no framework-specific keys found
            return 'vllm'

    @classmethod
    def create_job(
        cls,
        c_phdl,
        s_phdl,
        model_name,
        inference_config_dict,
        benchmark_params_dict,
        hf_token,
        gpu_type='mi300',
        distributed_inference=False,
        framework=None,
    ):
        """
        Create an inference job instance for the specified framework.

        Args:
            c_phdl: Client parallel handle
            s_phdl: Server parallel handle
            model_name: Name of the model
            inference_config_dict: Infrastructure configuration dictionary
            benchmark_params_dict: Benchmark parameters dictionary
            hf_token: HuggingFace token
            gpu_type: GPU type (default: 'mi300')
            distributed_inference: Whether to use distributed inference (default: False)
            framework: Framework type ('vllm', 'inferencemax', 'inferencex_atom', or None for auto-detect)

        Returns:
            A placeholder that raises — use suite jobs under ``cvs.tests.inference`` instead.

        Raises:
            ValueError: If framework is not supported
        """
        # Auto-detect framework if not specified
        if framework is None:
            framework = cls._detect_framework(inference_config_dict)
            log.info(f'Auto-detected framework: {framework}')

        framework_lower = framework.lower()
        if framework_lower == 'inferencemax':
            log.warning("framework='inferencemax' is deprecated; use inferencex_atom_single")
            framework_lower = 'inferencex_atom'

        if framework_lower not in cls._FRAMEWORK_CLASSES:
            supported = ', '.join(cls._FRAMEWORK_CLASSES.keys())
            raise ValueError(f"Unsupported framework: '{framework}'. Supported frameworks: {supported}")

        job_class = cls._FRAMEWORK_CLASSES[framework_lower]

        return job_class(
            c_phdl=c_phdl,
            s_phdl=s_phdl,
            model_name=model_name,
            inference_config_dict=inference_config_dict,
            benchmark_params_dict=benchmark_params_dict,
            hf_token=hf_token,
            gpu_type=gpu_type,
            distributed_inference=distributed_inference,
        )
