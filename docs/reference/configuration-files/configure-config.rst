.. meta::
  :description: Configure the details of each CVS configuration test file
  :keywords: configure, ROCm, test, health, RCCL, platform

************************
Test configuration files
************************

There are JSON configuration files for each CVS test. You must configure the JSON file for each test you want to run in CVS.

The test configuration files are in the ``cvs/input/config_file`` directory of the cloned repo.
You must go to each directory and edit the path given in the configuration file for each category. 

Here are code snippets and the parameters for each configuration file:

- :doc:`Platform </reference/configuration-files/platform>`
- :doc:`Health </reference/configuration-files/health>`
- :doc:`RCCL </reference/configuration-files/rccl>`
- :doc:`InfiniBand (IB Perf) </reference/configuration-files/ib>`
