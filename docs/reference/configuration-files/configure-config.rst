.. meta::
  :description: Configure the details of each CVS configuration test file
  :keywords: configure, ROCm, test, health, RCCL, platform

************************
Test configuration files
************************

Each CVS test has a corresponding JSON configuration file. You must configure the JSON file for each test you want to run in CVS.

The test configuration files are in the ``cvs/input/config_file`` directory of the cloned repo. You can go to each directory and edit the parameters as necessary for your testing requirements.

.. note::

  Ensure `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ is installed correctly, and the GPU drivers are loaded.

The following list provides a link to code snippets and the parameters for each configuration file:

- :doc:`Platform </reference/configuration-files/platform>`
- :doc:`Health </reference/configuration-files/health>`
- :doc:`InfiniBand (IB Perf) </reference/configuration-files/ib>`
- :doc:`RCCL </reference/configuration-files/rccl>`
- :doc:`JAX </reference/configuration-files/jax>`
- :doc:`Megatron </reference/configuration-files/megatron>`
- :doc:`MORI (RDMA Performance) </reference/configuration-files/mori>`
- :doc:`Aorta (Distributed Training) </reference/configuration-files/aorta>`
- :doc:`InferenceMAX (vLLM Benchmarking) </reference/configuration-files/inferencemax>`
- :doc:`vLLM Single-Node (MI355X) </reference/configuration-files/vllm_singlenode_mi355x>`
- :doc:`SGLang Disaggregated Prefill-Decode </reference/configuration-files/sglang_disagg_pd>`
- :doc:`Flux.1 Text-to-Image </reference/configuration-files/flux1_t2i>`
- :doc:`WAN 2.2 Image-to-Video </reference/configuration-files/wan22_i2v>`

Enable passwordless SSH
=======================

Passwordless SSH is enabled among the head and child nodes for the configuration files by default.

If Passwordless SSH is not enabled, use these commands to enable it (you can enable this for any SSH key, not just the RSA key):

.. tip:: 

  Perform these steps in reverse order (child node first, then head node) if you require a passwordless login from a head node to a child node.

1. Enable passwordless SSH for the head nodes:
   
   .. code:: bash
 
    cat ~/.ssh/id_rsa.pub

2. Enable passwordless SSH for the child nodes:
 
   .. code:: bash

    echo "paste-your-public-key-here" >> ~/.ssh/authorized_keys

   .. code:: bash

    chmod 600 ~/.ssh/authorized_keys
 
3. Then check these settings:

   .. code:: bash

    ssh username@remote_host-ip

4. If the username is also the same in both nodes, then use the IP address:

   .. code:: bash

    ssh remote-host-ip
