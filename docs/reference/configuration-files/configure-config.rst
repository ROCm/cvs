.. meta::
  :description: Configure the details of each CVS configuration test file
  :keywords: configure, ROCm, test, health, RCCL, platform

************************
Test configuration files
************************

Each CVS test has a corresponding JSON configuration file. You must configure the JSON file for each test you want to run in CVS.

The test configuration files are in the ``cvs/input/config_file`` directory of the cloned repo.
You must go to each directory and edit the path given in the configuration file for each category. 

The following list provides a link to code snippets and the parameters for each configuration file:

- :doc:`Platform </reference/configuration-files/platform>`
- :doc:`Health </reference/configuration-files/health>`
- :doc:`InfiniBand (IB Perf) </reference/configuration-files/ib>`
- :doc:`RCCL </reference/configuration-files/rccl>`
- :doc:`JAX </reference/configuration-files/jax>`
- :doc:`Megatron </reference/configuration-files/megatron>`