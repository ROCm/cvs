.. meta::
  :description: Run CVS MORI RDMA benchmark tests to check the SHMEM API and measure I/O throughput over InfiniBand fabric on AMD Instinct GPU cluster nodes.
  :keywords: CVS, MORI, RDMA, InfiniBand, AMD Instinct, ROCm, AMD, GPU, benchmark, network, IB, throughput

***********************************************
Run CVS MORI RDMA benchmark tests
***********************************************

MORI tests check the SHMEM API and measure RDMA I/O throughput using the MORI tree on the nodes. I/O cases run between nodes with ``torchrun``. Run MORI to validate raw RDMA performance before training or inference workloads.

.. _mori-set-up-config:

Prerequisites
=============

- A cluster file that lists the nodes for this run. The ``no_of_nodes`` field in the MORI config is not read. Node count is the number of nodes in the cluster file.
- For the shipped ``orchestrator`` value ``container``: Docker on the hosts, a GPU image that contains the MORI build, and ``/dev/dri`` plus ``/dev/kfd``. This value overrides the cluster file for this suite. Set ``orchestrator`` to ``baremetal`` to run on the host instead.
- Passwordless sudo on the hosts if you want the dmesg scan. The scan reads host ``dmesg`` on every node even when the benchmarks run in a container. ``test_verify_dmesg`` skips only when ``orch.sudo_prefix()`` is empty (which uses the head node's ``sudo -n true`` result).
- ``prettytable`` where ``benchmark.py`` runs. In a container, ``test_install_container_packages`` runs ``pip3 install prettytable``. On bare metal that test is skipped, so the host already needs ``prettytable``.

Set up config
=============

1. Copy the MORI configuration file:

   .. code:: bash

     cvs config copy mori/mi35x_mori_config.json --output ~/cvs_workspace/mori/mi35x_mori_config.json

2. Edit ``~/cvs_workspace/mori/mi35x_mori_config.json`` and replace every ``<changeme>`` before you run:

   - ``container.image`` — image that contains the MORI build
   - ``env.NCCL_SOCKET_IFNAME`` and ``env.GLOO_SOCKET_IFNAME`` — interface name exported into the benchmark
   - ``env.MORI_RDMA_DEVICES`` — comma-separated RDMA device names, checked against ``ibv_devinfo``
   - ``mori_dir`` — directory of the MORI tree (``tests/python/``)
   - ``master_addr`` — a host from the cluster file. That host is torchrun node rank 0 and the I/O initiator

   ``{user-id}`` in ``log_dir`` and in the home volume becomes the cluster file ``username`` (then ``$USER``, then ``root``). A leftover ``<changeme>`` exits at startup with ``Unresolved placeholder found in test config``.

   Keys named ``_example_options_orchestrator``, ``_example_env``, and ``_example_mori_dir`` are notes. ``_example_env`` is not copied into ``env``. Edit ``env`` itself.

3. Put the nodes in the cluster file. Changing ``no_of_nodes`` does not add or remove nodes.

For the field reference and the ``expected_results`` shape, see :doc:`/reference/configuration-files/network/mori`.

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
- ``shmem_api_$(hostname).log`` for the SHMEM test.

The shipped config mounts ``/home/{user-id}`` into the container and sets ``log_dir`` under that home, so a ``per_run`` container can exit and the logs remain on the host. ``test_teardown`` removes a ``per_run`` container. ``persistent`` and ``no_launch`` leave it running.

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

``test_cleanup_stale_containers`` always skips. Stale same-named containers are removed by a ``per_run`` launch. Dispatch and combine, concurrent-put, and IBGDA write tests are not in this suite.

Run on managed compute
======================

Follow :doc:`/how-to/run-on-managed-compute` to run MORI on managed compute. Set ``--workspace`` to storage shared by the nodes.
