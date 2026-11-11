.. meta::
  :description: Run CVS MORI RDMA benchmark tests to measure put, write, and I/O throughput over InfiniBand fabric on AMD Instinct GPU cluster nodes.
  :keywords: CVS, MORI, RDMA, InfiniBand, AMD Instinct, ROCm, AMD, GPU, benchmark, network, IB, throughput

***********************************************
Run CVS MORI RDMA benchmark tests
***********************************************

MORI tests measure RDMA put, write, and I/O throughput using the MORI tree in ``mori_dir`` on the nodes. ``install_mori`` builds that tree from git. I/O cases run between nodes with ``torchrun``. The IBGDA write case runs inside each node with ``mpiexec``. Run MORI to validate raw RDMA performance before training or inference workloads.

.. _mori-set-up-config:

Prerequisites
=============

- A cluster file that lists the nodes for this run. The ``no_of_nodes`` field in the MORI config is not read. Node count is the number of nodes in the cluster file.
- For the shipped ``orchestrator`` value ``container``: Docker on the hosts, a GPU image with ROCm (including ``hipcc``) and torch, and ``/dev/dri`` plus ``/dev/kfd``. This value overrides the cluster file for this suite. Set ``orchestrator`` to ``baremetal`` to run on the host instead.
- Passwordless sudo on the hosts if you want the dmesg scan. The scan reads host ``dmesg`` on every node even when the benchmarks run in a container. ``test_verify_dmesg`` skips only when ``orch.sudo_prefix()`` is empty (which uses the head node's ``sudo -n true`` result).
- ``prettytable`` where ``benchmark.py`` runs. In a container, ``test_install_container_packages`` runs ``pip3 install prettytable``. On bare metal, ``install_mori`` installs it into ``<mori_dir>/site``. If you skip ``install_mori`` on bare metal, the host needs ``prettytable``.

Set up config
=============

1. Copy the MORI configuration file:

   .. code:: bash

     cvs config copy mori/mi35x_mori_config.json --output ~/cvs_workspace/mori/mi35x_mori_config.json

2. Edit ``~/cvs_workspace/mori/mi35x_mori_config.json`` and replace every ``<changeme>`` before you run:

   - ``container.image`` — the GPU image the build and benchmarks run in
   - ``env.NCCL_SOCKET_IFNAME`` and ``env.GLOO_SOCKET_IFNAME`` — interface name exported into the benchmark
   - ``env.MORI_RDMA_DEVICES`` — comma-separated RDMA device names, checked against ``ibv_devinfo``
   - ``master_addr`` — a host from the cluster file. That host is torchrun node rank 0 and the I/O initiator

   ``mori_dir`` is where ``install_mori`` builds MoRI, ``/home/{user-id}/INSTALL/mori`` in the shipped file. To run a MoRI tree you built yourself, set ``mori_dir`` to it and skip ``install_mori``. ``git_ref`` picks the tag, branch, or commit to build.

   Delete the ``libbnxt_re-rdmav34.so`` and ``libibverbs.so.1`` volume lines if the nodes have no Broadcom Thor2 library. ``docker run`` fails when a listed host path is missing.

   ``{user-id}`` in ``mori_dir``, ``log_dir``, and the home volume becomes the cluster file ``username`` (then ``$USER``, then ``root``). A leftover ``<changeme>`` exits at startup with ``Unresolved placeholder found in test config``.

   Keys whose names start with ``_`` are notes. ``_example_env`` is not copied into ``env``. Edit ``env`` itself.

   In container mode, keep the shipped ``container.lifetime`` ``persistent``, so the container ``install_mori`` builds in is still there for the benchmark.

3. Put the nodes in the cluster file. Changing ``no_of_nodes`` does not add or remove nodes.

For the field reference and the ``expected_results`` shape, see :doc:`/reference/configuration-files/network/mori`.

.. _mori-install:

Build MoRI
==========

Run ``install_mori`` with the same files before ``mori_benchmark_test``, and again after changing ``git_ref``:

.. code:: bash

  cvs run install_mori \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mori/mi35x_mori_config.json \
    --workspace ~/cvs_workspace

- In container mode ``container.lifetime`` must be ``persistent`` or ``no_launch``. With ``per_run``, ``test_launch_mori_container`` fails at once, because the container and everything built in it would be removed when ``install_mori`` ends.
- MoRI is built into ``<mori_dir>/site`` and ``<mori_dir>/build``. The image's or host's own MoRI is not changed, and ``install_mori`` fails if the MoRI that ``python3`` finds already lives under ``mori_dir``.
- ``install_mori`` fetches ``git_url`` into ``mori_dir`` on every run, then checks out ``git_ref``. If the fetch fails, as on a node without network, it uses the existing checkout.
- When every node sees ``mori_dir``, the head node builds once. When each node has its own, every node builds.
- A missing build package fails the test, which lists it per node. With ``install_system_packages`` true, ``install_mori`` first apt-installs what it can: inside the container, or on bare metal with passwordless sudo. It never installs ROCm, torch, pip, or NIC libraries.
- Each building node writes ``<log_dir>/install_mori_<hostname>.log``, which starts with the MoRI commit. The benchmark's run card shows that commit.

.. _mori-run-tests:

Run tests
=========

The I/O cases are fixed in the suite. The config does not add or remove them. ``cvs list`` prints every name in alphabetical order, including the bracketed ids:

.. code:: bash

  cvs list mori_benchmark_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mori/mi35x_mori_config.json

The suite runs in this order:

.. code:: text

  test_cleanup_stale_containers          (always skipped)
  test_launch_mori_container             (skipped on bare metal)
  test_setup_ibv_devices
  test_install_container_packages        (skipped on bare metal)
  test_setup_env
  test_shmem_api
  test_concurrent_put_threads
  test_concurrent_put_imm_threads
  test_concurrent_put_signal_thread
  test_ibgda_write_test
  test_io_read[131072-256-1]
  test_io_read[131072-256-8]
  test_io_read[4194304-128-1]
  test_io_read[4194304-128-8]
  test_io_write[131072-256-1]
  test_io_write[131072-256-8]
  test_io_write[4194304-128-1]
  test_io_write[4194304-128-8]
  test_verify_dmesg
  test_teardown                          (skipped on bare metal)

The three numbers in an I/O id are the sweep maximum in bytes, the transfer batch size, and the QP count per transfer. Quote the id. Unquoted brackets are shell globs.

Run the suite with the file you edited:

.. code:: bash

  cvs run mori_benchmark_test \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mori/mi35x_mori_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv -s

Run one I/O case in the same process as container setup and teardown. Selecting only the bracketed id does not start the container:

.. code:: bash

  cvs run mori_benchmark_test \
    test_launch_mori_container \
    test_setup_ibv_devices \
    test_install_container_packages \
    test_setup_env \
    'test_io_read[4194304-128-1]' \
    test_verify_dmesg \
    test_teardown \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/mori/mi35x_mori_config.json \
    --workspace ~/cvs_workspace \
    --capture=tee-sys -vvv

On bare metal, launch, package install, and teardown skip. The I/O id still has to be quoted.

What a pass produces
====================

``cvs run`` writes these files under ``<workspace>/cvs_runs/<run_id>/``:

- ``mori_benchmark_test.html`` — pytest pass/fail.
- ``mori_benchmark_test_html/mori_benchmark_test_run_deck.html`` and ``.json`` — bandwidth and latency curves, linked from the pytest report as ``Mori Benchmark Run Deck``. The deck does not decide pass/fail.
- ``mori_benchmark_test.log`` — the pytest log.
- ``mori_benchmark_test_<timestamp>.zip`` — the pytest HTML, the run deck, and per-test logs.

MORI does not write a spreadsheet.

Benchmark logs are on the hosts under ``log_dir``, in a new directory named ``YYYYMMDD_HHMMSS``:

- ``io_<op>_<buffer>_<batch>_<qp>_node<rank>.log`` for each I/O case. Rank 0 is ``master_addr``.
- ``<name>_$(hostname).log`` for the SHMEM, concurrent-put, and IBGDA tests.

The shipped config mounts ``/home/{user-id}`` into the container and sets ``log_dir`` under that home, so the logs stay on the host after the container is gone. ``test_teardown`` removes a ``per_run`` container. ``persistent``, as shipped, and ``no_launch`` leave it running; remove a ``persistent`` container yourself when you are done.

Troubleshooting
===============

Search the pytest log for ``FAIL -``. The test then ends with ``Following FAILURES seen``.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - What you see
     - What it means
   * - ``Unresolved placeholder found in test config``
     - A ``<changeme>`` remains. Replace it in the file you copied.
   * - ``container.lifetime is per_run``
     - ``install_mori`` needs ``persistent`` or ``no_launch``.
   * - ``MoRI build dependencies missing on node``
     - The listed tools, headers, or libraries are absent on that node or in its container. See :ref:`mori-install` for what ``install_mori`` installs.
   * - ``is under mori_dir``
     - ``mori_dir`` points at the image's or host's own MoRI tree. Give it a directory of its own.
   * - ``mori_dir is not visible from node(s)``
     - Only some nodes share ``mori_dir``. Use a path that every node shares, or one that each node has locally.
   * - ``container.image is not set in the mori config``
     - ``container.image`` is empty. Launch did not start a container, and later benchmark tests skip with ``mori container launch failed``.
   * - ``setup_containers() returned False`` or ``not running after setup_containers()``
     - Docker did not keep the container up. Teardown still runs as the leak guard if ``test_teardown`` did not.
   * - ``env.MORI_RDMA_DEVICES is not set``
     - ``env.MORI_RDMA_DEVICES`` is missing or empty.
   * - ``MORI device <name> not showing up in ibv_devinfo``
     - That device name is not a substring of ``ibv_devinfo`` on that node.
   * - ``shmem test_api.py did not run properly`` or ``no PASSED``
     - ``pytest tests/python/shmem/test_api.py`` did not print ``PASSED``. A line containing ``FAIL`` also fails the test.
   * - ``concurrent_put_thread did not run properly``
     - The put and put-immediate binaries must print ``PASSED`` or ``test done!``. They do not check payload data. The signal-thread binary must print ``PASSED``.
   * - ``dist_write did not complete properly``
     - IBGDA output had no ``Index`` / ``Size`` header.
   * - ``IBGDA Mori BW less than expected``
     - The ``bw(GB)`` column was under ``expected_results.ibgda_write`` ``max_bw`` for that byte size. ``avg_lat`` in that block is not compared.
   * - ``master_addr <addr> is not one of the cluster hosts``
     - ``master_addr`` is not in the cluster file node list.
   * - ``MORI IO benchmark did not finish within 600s``
     - That host's log never gained ``MORI_IO_EXIT_CODE=``.
   * - ``MORI IO benchmark exited with code``
     - torchrun exited non-zero. The log path is in the message's node.
   * - ``no Bandwidth numbers seen``
     - The rank-0 log has no ``Max BW`` header, so the tables were not parsed.
   * - ``BW is lower than expected``
     - Average bandwidth (``Avg Bw (GB/s)``) was under ``max_bw`` for that rank and message size.
   * - ``Latency is higher than expected``
     - Average latency (``Avg Lat (us)``) was over ``avg_lat``.
   * - ``could not create log dir``
     - ``mkdir`` of ``log_dir/<timestamp>`` failed on that node.
   * - ``passwordless sudo unavailable on the hosts; dmesg scan needs sudo``
     - ``test_verify_dmesg`` was skipped because the head node's ``sudo -n true`` probe returned false (``orch.sudo_prefix()`` is empty).
   * - ``ERROR - Failue pattern``
     - Host dmesg matched a known error. The word ``Failue`` is the text CVS prints.

``test_cleanup_stale_containers`` always skips. Stale same-named containers are removed by a ``per_run`` launch. Dispatch and combine tests are not in this suite.

Run on managed compute
======================

Follow :doc:`/how-to/run-on-managed-compute` to run MORI on managed compute. Set ``--workspace`` to storage shared by the nodes.

Each ``cvs run`` is its own job step, so run ``install_mori`` and ``mori_benchmark_test`` in one allocation, or put ``mori_dir`` on storage the nodes share. A job step never runs apt, so the node image needs the build packages. If the compute nodes have no internet, clone ``git_url`` into a shared ``mori_dir`` beforehand.
