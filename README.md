# Cluster Validation Suite
CVS is a collection of tests scripts that can validate AMD AI clusters end to end from running single node burn in health tests to cluster wide distributed training and inferencing tests. CVS can be used by AMD customers to verify the health of the cluster as a whole which includes verifying the GPU/CPU node health, Host OS configuratin checks, NIC validations etc. CVS test suite collection comprises of the following set of tests

1. Platform Tests - Host OS config checks, BIOS checks, Firmware/Driver checks, Network config checks.
2. Burn in Health Tests - AGFHC, Transferbench, RocBLAS, rocHPL, Single node RCCL
3. Network Tests - Ping checks, Multi node RCCL validations for different collectives
4. Distributed Training Tests - Work in Progress
5. Distributed Inferencing Tests - Work in Progress

CVS leverages the PyTest open source framework to run the tests and generate reports and can be launched from a head-node or any linux management station which has connectivity to the cluster nodes via SSH. The single node tests are run in parallel cluster wide using the parallel-ssh open source python modules to optimize the time for running them. Currently CVS has been validated only on Ubuntu based Linux distro clusters. 
