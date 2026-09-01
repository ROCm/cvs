Scalability and Performance
===========================

CVS automatically scales to handle clusters from small lab setups to large enterprise deployments with thousands of nodes.

Parallel execution
------------------

CVS always runs cluster-wide SSH concurrently—it does not connect to hosts one at a time.

- **Default (up to 32 hosts)**: A single process uses gevent-based concurrent SSH (``ParallelSSHClient``) across all hosts.
- **Large host lists (more than 32 hosts, or ``CVS_HOSTS_PER_SHARD``)**: CVS additionally splits hosts into shards and runs each shard in a separate worker process. Each worker still uses gevent concurrency inside the process.
- **Results**: Output is merged while preserving the original host order.

**Example:** with 50 hosts and default settings, CVS:

1. Splits hosts into shards of 32
2. Runs each shard in a parallel worker process (gevent concurrency within each shard)
3. Merges results maintaining original host order

**Example:** with 8 hosts, CVS runs one gevent-based ``ParallelSSHClient`` over all hosts—no process sharding.

Environment Variables
---------------------

Configure CVS parallel SSH operations and optimize performance for your cluster size:

**CVS_HOSTS_PER_SHARD** (default: 32)
  When the host count exceeds this value, CVS splits work across multiple worker processes. Each process still uses gevent-based concurrent SSH. Lower the value to enable process sharding on smaller clusters; set ``0`` to disable process sharding (always one gevent client in the parent process).
  
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

**For smaller clusters or resource-constrained environments:**

.. code:: bash

  export CVS_HOSTS_PER_SHARD=16
  export CVS_WORKERS_PER_CPU=2

**Recommended Settings by Cluster Size:**

- **Large clusters (1000+ nodes)**: ``CVS_HOSTS_PER_SHARD=64``, ``CVS_WORKERS_PER_CPU=6-8``
- **Medium clusters (<1000 nodes)**: Default values (32 hosts per shard, 4 workers per CPU) usually work well
- **Small clusters (< 32 nodes)**: Defaults use single-process gevent concurrency; tune ``CVS_HOSTS_PER_SHARD`` only if you want process sharding at smaller host counts
- **Resource-constrained systems**: Lower both values to reduce memory and CPU usage