.. meta::
  :description: Configure the variables in the WAN 2.2 image-to-video configuration file
  :keywords: inference, ROCm, install, cvs, WAN, image-to-video, xDiT, MI300X

*************************************************
WAN 2.2 image-to-video configuration file
*************************************************

WAN 2.2 image-to-video tests validate distributed video generation performance using PyTorch xDiT on AMD GPU clusters. These tests ensure optimal performance for large-scale diffusion model inference converting images to videos.

The WAN 2.2 tests check:

- **Container orchestration**: Docker setup with ROCm and PyTorch xDiT for distributed video generation
- **Distributed inference**: Ulysses attention and Ring attention parallelization
- **Video generation**: Image-to-video synthesis with configurable parameters
- **Performance metrics**: Generation time, throughput, and video quality validation
- **Result verification**: Expected generation times and output quality

Change the parameters as needed in the WAN 2.2 configuration file: ``mi300x_wan22_i2v_a14b.json`` for image-to-video generation.

.. note::

  - ``{user-id}`` will be resolved to the current username in the runtime. You can also manually change this value to your username.
  - Replace all ``<changeme>`` placeholders with actual values for your cluster.

``mi300x_wan22_i2v_a14b.json``
===============================

Here's a code snippet of the ``mi300x_wan22_i2v_a14b.json`` file for reference:

.. dropdown:: ``mi300x_wan22_i2v_a14b.json``

  .. code:: json

    {
        "container_image": "rocm/pytorch:rocm6.2.4_ubuntu22.04_py3.10_pytorch_release_2.3.0",
        "container_name": "wan22_i2v_container",
        "model_id": "alibaba-pai/wan-2.2-14b-i2v",
        "hf_token_file": "/home/{user-id}/.hf_token",
        "input_image_path": "/home/{user-id}/INPUT/wan22_i2v/sample_image.png",
        "prompts": [
            "A serene mountain landscape with flowing water",
            "A bustling city street with people and vehicles"
        ],
        "height": 832,
        "width": 1216,
        "num_frames": 81,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "warmup_steps": 1,
        "log_dir": "/home/{user-id}/LOGS/wan22_i2v",
        "output_dir": "/home/{user-id}/OUTPUT/wan22_i2v",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {
                "/home/{user-id}": "/home/{user-id}",
                "/it-share/models": "/root/models"
            },
            "env_dict": {}
        },
        "expected_results": {
            "max_generation_time_seconds": 120,
            "min_videos_generated": 2
        }
    }

Parameters
==========

Use the parameters in this table to configure the WAN 2.2 image-to-video configuration file.

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
     - wan22_i2v_container
     - Name of the Docker container instance
   * - ``model_id``
     - alibaba-pai/wan-2.2-14b-i2v
     - HuggingFace model identifier for WAN 2.2 14B image-to-video model
   * - ``hf_token_file``
     - ``/home/{user-id}/`` |br| ``.hf_token``
     - Path to HuggingFace authentication token file
   * - ``input_image_path``
     - ``/home/{user-id}/`` |br| ``INPUT/wan22_i2v/`` |br| ``sample_image.png``
     - Path to input image for video generation
   * - ``prompts``
     - Array of text prompts
     - List of text prompts for video generation (e.g., ["A serene mountain landscape with flowing water", "A bustling city street with people and vehicles"])
   * - ``height``
     - 832
     - Height of generated video frames in pixels
   * - ``width``
     - 1216
     - Width of generated video frames in pixels
   * - ``num_frames``
     - 81
     - Number of video frames to generate
   * - ``num_inference_steps``
     - 50
     - Number of denoising steps for video generation
   * - ``guidance_scale``
     - 6.0
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
     - ``/home/{user-id}/`` |br| ``LOGS/wan22_i2v``
     - Directory for inference logs
   * - ``output_dir``
     - ``/home/{user-id}/`` |br| ``OUTPUT/wan22_i2v``
     - Directory for generated videos
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
     - 120
     - Maximum expected time for video generation in seconds
   * - ``expected_results.`` |br| ``min_videos_generated``
     - 2
     - Minimum number of videos expected to be generated successfully
