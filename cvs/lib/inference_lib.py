'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

from cvs.lib.inference_max_lib import InferenceMaxJob, VllmJob


class InferenceJobFactory:
    """Factory class to create inference job instances based on framework type."""

    # Registry of supported frameworks
    _FRAMEWORK_CLASSES = {
        'vllm': VllmJob,
        'inferencemax': InferenceMaxJob,
    }

    @classmethod
    def _detect_framework(cls, inference_config_dict):
        """
        Auto-detect framework type from inference config dictionary.

        Args:
            inference_config_dict: Infrastructure configuration dictionary

        Returns:
            Detected framework name ('vllm' or 'inferencemax')

        Detection logic:
            - If 'inferencemax_repo' is present → InferenceMAX
            - If 'vllm_script_path' is present → vLLM
            - Otherwise → vLLM (default)
        """
        if 'inferencemax_repo' in inference_config_dict:
            return 'inferencemax'
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
            framework: Framework type ('vllm', 'inferencemax', or None for auto-detect)

        Returns:
            Instance of VllmJob or InferenceMaxJob

        Raises:
            ValueError: If framework is not supported

        Examples:
            # Auto-detect from config:
            job = InferenceJobFactory.create_job(
                c_phdl=client_handle,
                s_phdl=server_handle,
                model_name='qwen3-80b',
                inference_config_dict=config,  # Contains 'inferencemax_repo' or 'vllm_script_path'
                benchmark_params_dict=params,
                hf_token=token
            )

            # Explicit framework:
            job = InferenceJobFactory.create_job(
                framework='vllm',
                c_phdl=client_handle,
                s_phdl=server_handle,
                model_name='qwen3-80b',
                inference_config_dict=config,
                benchmark_params_dict=params,
                hf_token=token
            )
        """
        # Auto-detect framework if not specified
        if framework is None:
            framework = cls._detect_framework(inference_config_dict)
            print(f'Auto-detected framework: {framework}')

        framework_lower = framework.lower()

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
