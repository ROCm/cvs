.. meta::
  :description: Run the Cluster Health Checker utility script to generate a health report of your GPU cluster
  :keywords: CVS, health, network, tests, RCCL 

**********************************
Monitor the health of GPU clusters
**********************************

Monitor the health of your cluster with a standalone Python utility script that generates an overall health report by collecting logs and metrics of the GPU nodes.

The Cluster Health Checker utility script identifies any hardware failure/degradation signatures like RAS errors, PCIe/XGMI errors, network drop / error counters. 
It can also identify software failures by searching for failuring signatures in the ``demsg`` and ``journlctl`` logs.

The script also acts as a triaging tool to troubleshoot any performance issues that may be related to the AI infrastructure. 
You can take a snapshot of all counters (GPU/NIC) while your training/inference workloads are in progress, 
then compare the counters and identify any increment of unexpected counters across all nodes in the cluster to find issues.

Generate a health report
========================

Run the Cluster Health Checker utility script to generate a health report for your clusters. 



