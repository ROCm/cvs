Scalability and Performance
===========================

CVS automatically scales to handle clusters from small lab setups to large enterprise deployments with thousands of nodes.

⚠️ Important Behavioral Change
------------------------------

**Starting with this version, CVS automatically uses multi-process parallel execution for improved performance:**

- **When it activates**: Automatically when host lists exceed 32 nodes (or ``CVS_HOSTS_PER_SHARD`` value)
- **What changes**: SSH operations are distributed across multiple processes instead of running sequentially
- **Performance impact**: Significantly faster execution for large clusters
- **Compatibility**: All existing code continues to work - just runs more efficiently
- **Migration**: No code changes required, but be aware of the different execution pattern

**Example of the behavioral change:**

.. code:: bash

  # With 50 hosts, CVS now automatically:
  # 1. Splits hosts into shards of 32 (default)
  # 2. Processes each shard in parallel using multiple worker processes
  # 3. Merges results maintaining original host order
  
  cvs run health_multinode_cvs --cluster_file cluster.json --config_file health_config.json

Environment Variables
---------------------

Configure CVS parallel SSH operations and optimize performance for your cluster size:

**CVS_HOSTS_PER_SHARD** (default: 32)
  Controls how many hosts are processed in each parallel shard. When running commands across many hosts, CVS automatically splits the host list into smaller chunks for efficient parallel processing.
  
  .. code:: bash
  
    export CVS_HOSTS_PER_SHARD=64  # Process 64 hosts per shard instead of default 32

**CVS_WORKERS_PER_CPU** (default: 4)
  Sets the number of worker processes per CPU core for parallel operations. The total number of workers is calculated as ``CPU_COUNT * CVS_WORKERS_PER_CPU``.
  
  .. code:: bash
  
    export CVS_WORKERS_PER_CPU=8  # Use 8 workers per CPU core instead of default 4

Performance Tuning Examples
---------------------------

**For large clusters (1000+ nodes):**

.. code:: bash

  export CVS_HOSTS_PER_SHARD=64
  export CVS_WORKERS_PER_CPU=6
  cvs run health_multinode_cvs --cluster_file cluster.json --config_file health_config.json

**For smaller clusters or resource-constrained environments:**

.. code:: bash

  export CVS_HOSTS_PER_SHARD=16
  export CVS_WORKERS_PER_CPU=2
  cvs run platform_multinode_cvs --cluster_file cluster.json --config_file platform_config.json

**Recommended Settings by Cluster Size:**

- **Large clusters (1000+ nodes)**: ``CVS_HOSTS_PER_SHARD=64``, ``CVS_WORKERS_PER_CPU=6-8``
- **Medium clusters (<1000 nodes)**: Default values (32 hosts per shard, 4 workers per CPU) usually work well
- **Small clusters (< 32 nodes)**: ``CVS_HOSTS_PER_SHARD=8``, ``CVS_WORKERS_PER_CPU=2`` # to increase parallelism
- **Resource-constrained systems**: Lower both values to reduce memory and CPU usage