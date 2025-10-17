.. meta::
  :description: CVS is a collection of test scripts that can validate AMD AI clusters
  :keywords: CVS, ROCm, documentation, test scripts, validation

What is Cluster Validation Suite (CVS)?
=======================================

CVS is a collection of test scripts that validate AMD AI clusters. 
Use CVS to verify cluster health, GPU/CPU node health, Host OS configuration checks, and NIC validations.

Here are the tests available in the CVS:

-	**Platform tests**: Perform host OS configuration, BIOS, firmware/driver, and network configuration checks.
-	**Burn in health tests**: Perform AMD GPU Field Health Check (AGFHC), Transferbench, RocBLAS, rocHPL, and Babelstream tests.
-	**Network tests**: Perform ping checks and multi-node ROCm Communication Collectives Library (RCCL) validations for different collectives.
- **InfiniBand (IB Perf)**: These tests are low-level network performance benchmarks that validate the raw communication capabilities of InfiniBand adapters and interconnects. These tests measure the fundamental building blocks that RCCL and other high-level libraries depend on.

CVS uses the PyTest open source framework to run the tests and generate reports. It can be launched from a head node or any Linux management station which has connectivity to the cluster nodes via SSH. 
The single node tests are run in parallel (cluster wide) using the parallel-SSH open source Python modules to optimize their running time. 

.. note::

   CVS has only been validated on Ubuntu-based Linux distro clusters.
