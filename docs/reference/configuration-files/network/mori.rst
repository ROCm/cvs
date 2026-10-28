.. meta::
  :description: Reference for the MORI network configuration file in CVS, covering RDMA read/write bandwidth, latency thresholds, container setup, and NIC device options.
  :keywords: CVS, MORI, RDMA, ROCm, network, bandwidth, latency, InfiniBand, AMD, GPU, JSON, configuration, multi-node

****************************************************************************
MORI network configuration file reference for Cluster Validation Suite (CVS)
****************************************************************************

MORI (Memory-Oriented RDMA Interface) benchmarks run from ``mori_dir`` through the orchestrator. The shipped file is ``input/config_file/mori/mi35x_mori_config.json``.

The suite checks:

- Container or bare-metal launch, from ``orchestrator`` and ``container``
- That every name in ``env.MORI_RDMA_DEVICES`` appears in ``ibv_devinfo``
- The fixed I/O matrix
- Bandwidth and latency against ``expected_results``, for the keys the checker understands
- Host dmesg over the suite window, gated on the head node's passwordless sudo probe

.. note::

  - ``{user-id}`` is replaced with the cluster file ``username``, then ``$USER``, then ``root``. ``{user}``, ``{home}``, ``{home-mount-dir}``, ``{node-dir-name}``, and ``{run_dir}`` are replaced in the same pass.
  - Any string that still contains ``<changeme>`` stops the run with ``Unresolved placeholder found in test config``.
  - A key whose name starts with ``_`` is not a setting. ``_example_env`` is not merged into ``env``.

``mi35x_mori_config.json``
==========================

The block below is the shipped file with the ``_example_*`` notes removed. Those notes are still in the file on disk and are not applied.

.. dropdown:: ``mi35x_mori_config.json``

  .. code:: json

    {
        "gpu_name": "mi35x",
        "no_of_nodes": "2",
        "orchestrator": "container",
        "container": {
            "image": "<changeme>",
            "name": "mori_container",
            "lifetime": "per_run",
            "runtime": {
                "name": "docker",
                "args": {
                    "devices": [ "/dev/dri", "/dev/kfd" ],
                    "volumes": [ "/home/{user-id}:/home/{user-id}" ]
                }
            }
        },
        "env": {
            "NCCL_SOCKET_IFNAME": "<changeme>",
            "GLOO_SOCKET_IFNAME": "<changeme>",
            "MORI_RDMA_DEVICES": "<changeme>",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"
        },
        "mori_dir": "<changeme>",
        "master_addr": "<changeme>",
        "master_port": "1234",
        "log_dir": "/home/{user-id}/LOGS/mori",
        "expected_results": {
            "ibgda_write": {
                "PROCS:2,CTAS:2,THREADS:256,QP_COUNT:4": {
                    "2097152": { "max_bw": "46.0", "avg_lat": "24.4" },
                    "4194304": { "max_bw": "46.0", "avg_lat": "48.8" }
                }
            },
            "io_read": {
                "BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:1": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2920" }
                },
                "BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:8": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2920" }
                }
            },
            "io_write": {
                "BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:1": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2920" }
                },
                "BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:8": {
                    "524288": { "max_bw": "45.0", "avg_lat": "1500" },
                    "1048576": { "max_bw": "46.0", "avg_lat": "2920" }
                }
            }
        }
    }

Parameters the run uses
=======================

The MORI config's ``container`` object is deep-merged over the cluster file's ``container``. ``orchestrator`` in this file overrides the cluster file when it is set.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Parameter
     - Shipped value
     - What it changes
   * - ``orchestrator``
     - ``container``
     - ``container`` runs the benchmarks in Docker. ``baremetal`` runs them on the host. Omitted, the cluster file's orchestrator is used, and that defaults to bare metal.
   * - ``container.image``
     - ``<changeme>``
     - Image to launch. Empty image fails ``test_launch_mori_container`` with ``container.image is not set``.
   * - ``container.name``
     - ``mori_container``
     - Container name. A ``per_run`` launch removes a container of this name before starting.
   * - ``container.lifetime``
     - ``per_run``
     - ``per_run`` starts a container and force-removes it at teardown. ``persistent`` attaches when that name is already running on every node, and starts one when it is absent on every node. A mix of running and absent nodes fails setup. ``no_launch`` only checks that the named container is already running.
   * - ``container.runtime.name``
     - ``docker``
     - Container runtime. Omitted, the runtime is Docker.
   * - ``container.runtime.args.devices``
     - ``/dev/dri``, ``/dev/kfd``
     - Extra devices, merged with the orchestrator defaults ``/dev/kfd`` and ``/dev/dri``. A Docker launch also passes each host's ``/dev/infiniband/*`` nodes.
   * - ``container.runtime.args.volumes``
     - ``/home/{user-id}:/home/{user-id}``
     - Extra ``host:container`` mounts, appended after ``/home/<user running cvs>/.ssh:/host_ssh``. Add a mount here when the image needs a host library.
   * - ``env``
     - See the sample
     - Exported on every benchmark command. This is separate from ``container.env``. ``MORI_RDMA_DEVICES`` is the device list checked in ``ibv_devinfo``. ``LD_LIBRARY_PATH`` may append ``$LD_LIBRARY_PATH``. ``NCCL_SOCKET_IFNAME`` and ``GLOO_SOCKET_IFNAME`` are exported the same way.
   * - ``mori_dir``
     - ``<changeme>``
     - Working directory. I/O runs ``<mori_dir>/tests/python/io/benchmark.py``. Required.
   * - ``master_addr``
     - ``<changeme>``
     - Torchrun node rank 0 and the I/O initiator. It must be one of the cluster hosts. Omitted, the cluster head node is used.
   * - ``master_port``
     - ``1234``
     - Torchrun rendezvous port. Required.
   * - ``log_dir``
     - ``/home/{user-id}/LOGS/mori``
     - Parent of the per-run log directory ``<log_dir>/YYYYMMDD_HHMMSS``. Required.
   * - ``expected_results``
     - See below
     - Thresholds. Required once a benchmark test builds ``MoriBenchmark``. The lookup shape is below.
   * - ``gpu_name``
     - ``mi35x``
     - Shown on the HTML run card only. It does not change binaries, devices, or thresholds.

Shipped keys the checker does not read
=======================================

- ``no_of_nodes`` is in the shipped file and is not read. The node list is the cluster file.
- ``expected_results.ibgda_write`` is in the shipped file and is not read. The IBGDA write test does not run.
- ``gpu_name`` is not used as a threshold or a device filter.
- ``_example_options_orchestrator``, ``_example_env``, and ``_example_mori_dir`` are not applied. ``_example_mori_dir`` (``/sgl-workspace/mori``) is an illustration in the file, not the path the suite opens.
- There is no ``nic_type``, ``oob_port``, ``mori_device_list``, ``torchlib_dir``, or ``container_config`` key. Devices are ``env.MORI_RDMA_DEVICES``. The out-of-band interface is whatever you export in ``env``.

``expected_results``
====================

The checker looks up two top-level names: ``io_read`` and ``io_write``. A missing case key does not fail the test. That case runs and is not gated.

I/O
---

The case key is one string:

.. code:: text

  BUFF_SIZE:<buffer bytes>,TRANSFER_SIZE:<batch>,QP_COUNT:<qp>

``TRANSFER_SIZE`` is the transfer batch count, not a byte size. Under that key, each message size in bytes maps to ``max_bw`` and ``avg_lat``.

For a printed table row whose byte size matches a key:

- Bandwidth fails when ``Avg Bw (GB/s)`` is lower than ``max_bw``. The column is the average. The checker compares that average to the value stored under ``max_bw``. An equal value passes. Units are GB/s.
- Latency fails when ``Avg Lat (us)`` is higher than ``avg_lat``. Units are microseconds.

The failure lines are ``BW is lower than expected`` and ``Latency is higher than expected``, with the rank, message size, and the case key.

The I/O matrix is fixed. Each case sweeps 8 bytes, doubling, up to the buffer size, with session mode, 8 initiator devices, and 8 target devices. Those device counts are not config keys.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Pytest id
     - Case key the checker builds
     - Shipped sample
   * - ``test_io_read[131072-256-1]`` and ``test_io_write[131072-256-1]``
     - ``BUFF_SIZE:131072,TRANSFER_SIZE:256,QP_COUNT:1``
     - No entry. The sample does not gate these cases.
   * - ``test_io_read[131072-256-8]`` and ``test_io_write[131072-256-8]``
     - ``BUFF_SIZE:131072,TRANSFER_SIZE:256,QP_COUNT:8``
     - No entry. The sample does not gate these cases.
   * - ``test_io_read[4194304-128-1]`` and ``test_io_write[4194304-128-1]``
     - ``BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:1``
     - Gates 524288 and 1048576 bytes only.
   * - ``test_io_read[4194304-128-8]`` and ``test_io_write[4194304-128-8]``
     - ``BUFF_SIZE:4194304,TRANSFER_SIZE:128,QP_COUNT:8``
     - Gates 524288 and 1048576 bytes only.

A size that is in ``expected_results`` but never appears in the table is not a failure. Only a printed row is compared. The shipped numbers are what the checker uses for those keys. Replace them for your fabric. The suite has no second threshold table.

Current behavior
================

- ``test_cleanup_stale_containers`` always skips. A ``per_run`` launch removes a same-named container itself.
- ``test_verify_dmesg`` scans host dmesg from orchestrator creation to the end of that test (second resolution, closed window) with ``verify_dmesg_for_errors``. The test skips only when ``orch.sudo_prefix()`` is empty (taken from the head node's ``sudo -n true`` probe).
- A Docker launch also applies orchestrator defaults: ``privileged`` true, ``network`` ``host``, ``ipc`` ``host``, capabilities ``SYS_PTRACE``, ``IPC_LOCK``, and ``SYS_ADMIN``, group ``video``, ulimit ``memlock=-1``, security options ``seccomp=unconfined`` and ``apparmor=unconfined``, and container environment ``GPUS=8`` and ``MULTINODE=true``. ``container.runtime.args.network``, ``ipc``, and ``privileged`` replace those three. ``cap_add``, ``security_opt``, ``group_add``, and ``ulimit`` are appended to the defaults. ``container.env`` updates the container environment. The top-level ``env`` object is only the benchmark command prefix.
- A failed container launch sets the rest of the benchmark tests to skip (``mori container launch failed``). ``test_teardown`` still runs.
- I/O waits up to 600 seconds for ``MORI_IO_EXIT_CODE=`` on every host, then kills leftover ``benchmark.py`` and torchrun processes.
