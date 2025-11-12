.. meta::
  :description: CVS is a collection of test scripts that can validate AMD AI clusters
  :keywords: CVS, ROCm, documentation, test scripts, validation

What is Cluster Validation Suite (CVS)?
=======================================

CVS is a collection of test scripts that validate AMD AI clusters. 
Use CVS to verify GPU cluster health, GPU/CPU node health, host OS configuration, and NIC (network interface card) validation.

Here are the tests available in the CVS:

-	**Platform tests**: Perform host OS configuration, BIOS, firmware/driver, and network configuration checks.
-	**Burn-in health tests**: Perform `AMD GPU Field Health Check (AGFHC) <https://instinct.docs.amd.com/projects/gpu-operator/en/latest/test/agfhc.html>`_, `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/install/install.html#install-transferbench>`_, and `ROCm Validation Suite (RVS) <https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/install/installation.html>`_.
- **InfiniBand (IB Perf)**: These tests are low-level network performance benchmarks that validate the raw communication capabilities of InfiniBand adapters and interconnects. These tests measure the fundamental building blocks on which RCCL and other high-level libraries depend.
-	**Network tests**: Perform ping checks and multi-node `ROCm Communication Collectives Library (RCCL) <https://rocm.docs.amd.com/projects/rccl/en/latest/install/installation.html>`_ validations for different collectives.
- **Distributed training tests**: Run and validate Llama 3.1 70B and 405B model distributed trainings across a multi-node cluster with the `JAX <https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html>`_ and `Megatron <https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/stanford-megatron-lm-compatibility.html>`_ frameworks. 
  
  - The JAX training file uses PyTest and parallel SSH to prepare the environment, launch containers, and run/verify a short distributed training job.
  - Megatron training enables scaling transformer models from millions to trillions of parameters by efficiently utilizing hundreds or thousands of GPUs across multiple nodes.

You can also :doc:`Monitor the health of GPU clusters <how-to/run-cluster>` using the Cluster Health Checker utility script. 
This script generates an overall health report that you can use to diagnose issues in your cluster.

CVS uses the open-source PyTest framework to run the tests and generate reports. You can launch CVS from a head node or any Linux management station that has connectivity to the cluster nodes via SSH. 
The single node tests run cluster-wide in parallel using the open-source parallel-SSH Python modules to optimize their running time.  

.. note::

   CVS has been validated on Ubuntu-based Linux distribution clusters.
