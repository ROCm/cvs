.. meta::
  :description: Configure the variables in the Flux.1-dev text-to-image configuration file
  :keywords: inference, ROCm, install, cvs, Flux, text-to-image, xDiT, MI300X

*************************************************
Flux.1-dev text-to-image configuration file
*************************************************

Flux.1-dev text-to-image tests validate distributed image generation performance using PyTorch xDiT on AMD GPU clusters. These tests ensure optimal performance for large-scale diffusion model inference with efficient parallelization strategies.

The Flux.1-dev tests check:

- **Container orchestration**: Docker setup with ROCm and PyTorch xDiT for distributed inference
- **Distributed inference**: Ulysses attention and Ring attention parallelization
- **Image generation**: Text-to-image synthesis with configurable parameters
- **Performance metrics**: Generation time, throughput, and image quality validation
- **Result verification**: Expected generation times and output quality

Change the parameters as needed in the Flux.1-dev configuration file: ``mi300x_flux1_dev_t2i.json`` for text-to-image generation.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - Replace all ``<changeme>`` placeholders with actual values for your cluster.

``mi300x_flux1_dev_t2i.json``
==============================

Here's a code snippet of the ``mi300x_flux1_dev_t2i.json`` file for reference:

.. dropdown:: ``mi300x_flux1_dev_t2i.json``

  .. code:: json

    {
        "container_image": "rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0",
        "container_name": "flux1_dev_t2i_container",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "hf_token_file": "/home/{user-id}/.hf_token",
        "prompts": [
            "A cat holding a sign that says hello world",
            "A futuristic cityscape at sunset with flying cars"
        ],
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "warmup_steps": 1,
        "log_dir": "/home/{user-id}/LOGS/flux1_dev_t2i",
        "output_dir": "/home/{user-id}/OUTPUT/flux1_dev_t2i",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {
                "/home/{user-id}": "/home/{user-id}",
                "/it-share/models": "/root/models"
            },
            "env_dict": {}
        },
        "expected_results": {
            "max_generation_time_seconds": 45,
            "min_images_generated": 2
        }
    }

Parameters
==========

Use the parameters in this table to configure the Flux.1-dev text-to-image configuration file.

.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameters
     - Default values
     - Description
   * - ``container_image``
     - rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0
     - Docker container image with ROCm and PyTorch for inference
   * - ``container_name``
     - flux1_dev_t2i_container
     - Name of the Docker container instance
   * - ``model_id``
     - black-forest-labs/FLUX.1-dev
     - HuggingFace model identifier for Flux.1-dev
   * - ``hf_token_file``
     - ``/home/{user-id}/`` |br| ``.hf_token``
     - Path to HuggingFace authentication token file
   * - ``prompts``
     - Array of text prompts
     - List of text prompts for image generation (e.g., ["A cat holding a sign that says hello world", "A futuristic cityscape at sunset with flying cars"])
   * - ``height``
     - 1024
     - Height of generated images in pixels
   * - ``width``
     - 1024
     - Width of generated images in pixels
   * - ``num_inference_steps``
     - 28
     - Number of denoising steps for image generation
   * - ``guidance_scale``
     - 3.5
     - Guidance scale for classifier-free guidance (higher values follow prompt more closely)
   * - ``ulysses_degree``
     - 2
     - Ulysses attention parallelization degree (splits sequence length across GPUs)
   * - ``ring_degree``
     - 1
     - Ring attention parallelization degree (splits across GPUs with ring communication)
   * - ``warmup_steps``
     - 1
     - Number of warmup inference steps before measurement
   * - ``log_dir``
     - ``/home/{user-id}/`` |br| ``LOGS/flux1_dev_t2i``
     - Directory for inference logs
   * - ``output_dir``
     - ``/home/{user-id}/`` |br| ``OUTPUT/flux1_dev_t2i``
     - Directory for generated images
   * - ``container_config.`` |br| ``device_list``
     - Values: |br| - ``"/dev/dri"`` |br| - ``"/dev/kfd"``
     - List of device paths to mount in the container for GPU access
   * - ``container_config.`` |br| ``volume_dict``
     - Multiple mappings
     - Dictionary mapping host paths to container paths for volume mounts
   * - ``/home/{user-id}``
     - ``/home/{user-id}``
     - User home directory mount
   * - ``/it-share/models``
     - ``/root/models``
     - Models directory mount
   * - ``container_config.`` |br| ``env_dict``
     - Empty
     - Dictionary of environment variables to set in the container
   * - ``expected_results.`` |br| ``max_generation_time_`` |br| ``seconds``
     - 45
     - Maximum expected time for image generation in seconds
   * - ``expected_results.`` |br| ``min_images_generated``
     - 2
     - Minimum number of images expected to be generated successfully
