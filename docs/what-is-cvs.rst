.. meta::
  :description: CVS is a collection of test scripts that can validate AMD AI clusters
  :keywords: CVS, ROCm, documentation, test scripts, validation

What is Cluster Validation Suite (CVS)?
=======================================

CVS is a collection of test scripts that validate AMD AI clusters. 
Use CVS to verify cluster health, GPU/CPU node health, host OS configuration, and NIC (network interface card) validation.

Here are the tests available in the CVS:

-	**Platform tests**: Perform host OS configuration, BIOS, firmware/driver, and network configuration checks.
-	**Burn-in health tests**: Perform AMD GPU Field Health Check (AGFHC), `TransferBench <https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html>`_, `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_, and Babelstream tests.
-	**Network tests**: Perform ping checks and multi-node `ROCm Communication Collectives Library (RCCL) <https://rocm.docs.amd.com/projects/rccl/en/latest/index.html>`_ validations for different collectives.
- **InfiniBand (IB Perf)**: These tests are low-level network performance benchmarks that validate the raw communication capabilities of InfiniBand adapters and interconnects. These tests measure the fundamental building blocks on which RCCL and other high-level libraries depend.

CVS uses the open-source PyTest framework to run the tests and generate reports. You can launch CVS from a head node or any Linux management station that has connectivity to the cluster nodes via SSH. 
The single-node tests run cluster-wide in parallel using the open-source parallel-SSH Python modules to optimize their running time.  

.. note::

   CVS has been validated on Ubuntu-based Linux distribution clusters.
