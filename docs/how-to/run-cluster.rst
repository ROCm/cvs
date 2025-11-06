.. meta::
  :description: Run the Cluster Health Checker utility script to generate a health report of your GPU cluster
  :keywords: CVS, health, network, tests, RCCL 

**********************************
Monitor the health of GPU clusters
**********************************

Monitor the health of your cluster with the Cluster Health Checker utility script (``check_cluster_health.py`` in the ``utils`` folder of the CVS GitHub repo), a standalone Python utility script that generates an overall health report by collecting logs and metrics of the GPU nodes.
The script doesn't require any agent/plugin/exporters to be installed or any controller virtual machines. It can provide deep visibilty against any cluster (such as a slurm cluster or Kubernetes cluster).

The script identifies any hardware failure/degradation signatures like RAS errors, PCIe/XGMI errors, or network drop / error counters. 
It can also identify software failures by searching for failing signatures in the ``demsg`` and ``journlctl`` logs.

The script also acts as a triaging tool to troubleshoot any performance issues that may be related to the AI infrastructure. 
You can take a snapshot of all counters (GPU/NIC) while your training/inference workloads are in progress, 
then compare the counters and identify any increment of unexpected counters across all nodes in the cluster to find issues.

Generate a health report
========================

Run the Cluster Health Checker utility script to generate a health report for your clusters with Python commands. 

To run the script and generate a health report for a cluster:

1. Ensure you've completed the :doc:`Cluster Validation Suite installation </install/cvs-install>`.
2. Open a new Terminal and CD into the cloned ``cvs`` repo.
3. Type this Python command:

   .. code:: Python

    python3 ./utils/check_cluster_health.py 

   Then set the applicable arguments for your use case:

   - ``--hosts``: Direct the script to the file with the list of host IP addresses you want the script to check.
   - ``--username``: Enter the username to SSH to the hosts.
   - ``--password``: Enter the password to SSH to the hosts
   - ``--key_file ``: Enter the private Keyfile for the username.
   - ``--iterations``: Enter the number of check iterations you want to run.
   - ``--time_between_iters``: Enter the time the script should wait between run iterations.
   - ``--report_file``: Enter the directory you want the generated health file to save to. If you leave this argument empty, the file saves as ``cluster_report.html`` to the local directory.  

   Here's an example command with some arguments set:

   .. code:: Python

    python3 ./utils/check_cluster_health.py --hosts /home/user/input/host_file.txt --username myusername --key_file /home/user/input/.ssh/id_test --iterations 2

   The script logs into the nodes based on the hosts specified and captures information on potential error conditions or anomalies. 

4. Open the ``cluster_report.html`` file to view the generated health report for the cluster.

Review the health report
========================

Open the generated health report to view snapshotted information on your cluster such as the:

- Cluster summary
- GPU information
- NIC information
- Historic error logs
- Snapshot differences

The script uses the `AMD SMI Python library <https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html>` 
to scan and capture the GPU cluster information. 

It looks for any potential errors, then graphs them in tables separated by categories such as PCIe errors, RDMA statistics, network congestion errors, GPU errors, or GPU cable issues. 
Detected anomalies are highlighted in red:

.. image:: ../images/rdma.png

.. image:: ../images/pcie.png

The report also displays potential kernel error in the ``dmesg`` and ``journlctl`` logs:

.. image:: ../images/journlctl.png












