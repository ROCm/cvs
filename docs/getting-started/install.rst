.. meta::
  :description: Component install 
  :keywords: Component, ROCm, install

*******
Install
*******

System requirements
===================

CVS supports these GPUs:

- AMD Instinct™ MI325X
- AMD Instinct™ MI300X	
- AMD Instinct™ MI350X
- AMD Instinct™ MI355X
- AMD Instinct™ MI455X

CVS supports these Linux distributions:

.. list-table::
   :widths: 3 3 4 4
   :header-rows: 1

   * - Operating system
     - Kernel
     - ROCm version (tested on)
     - Python version (tested on)
   * - Ubuntu 24.04.3
     - 6.8 [GA], 6.14 [HWE]
     - 7.0.2
     - 3.10
   * - Ubuntu 22.04.5
     - 5.15 [GA], 6.8 [HWE]
     - 7.0.2
     - 3.10

Install CVS
===========

Run CVS from a head node — an Ubuntu VM or bare-metal machine, with or without a GPU.
It is recommended to use a head node that is **not** part of the test cluster, so a reboot or failure on a worker does not take out your control plane.

Two installation options
------------------------

You can install and run the CVS CLI in either of these ways:

#. **Python virtual environment** — install CVS with ``pip`` into a venv on the head node (``make install`` or manual setup). This is the usual choice for development and day-to-day use. See `Install in a Python virtual environment`_.

#. **Docker container** — build the CVS image and run ``cvs`` inside a container on the head node. The container connects to cluster nodes over SSH; it does not include ROCm or the workloads CVS launches on workers. See `Install and run in a Docker container`_.

In both cases CVS orchestrates tests on remote cluster nodes over SSH. The install location only affects where the **CLI** runs.

Install in a Python virtual environment
---------------------------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.9 or later
- Git

Debian/Ubuntu Systems
~~~~~~~~~~~~~~~~~~~~~~

On Debian and Ubuntu distributions, the ``venv`` module is not included in the base Python package. Install it before proceeding:

.. code:: bash

  sudo apt install python3-venv

Two installation methods
~~~~~~~~~~~~~~~~~~~~~~~~

Within a Python virtual environment, you can install CVS in either of these ways:

#. **Makefile** — run ``make install`` from the repository root. CVS builds the package, creates ``.cvs_venv/``, and installs into it. This is the fastest path for most users. See `Install with Makefile`_.

#. **pip** — build a source distribution with ``python setup.py sdist``, create your own venv, and ``pip install`` the tarball. Use this when you need a custom venv name or location. See `Install with pip`_.

Install with Makefile
~~~~~~~~~~~~~~~~~~~~~

This is the quickest way to install CVS from source.

1. Clone the repository and install using make:

   .. code:: bash

     git clone https://github.com/ROCm/cvs
     cd cvs
     make install

   This will automatically:

   - Build the source distribution
   - Create a virtual environment in ``.cvs_venv/``
   - Install CVS in the virtual environment

2. Activate the virtual environment:

   .. code:: bash

     source .cvs_venv/bin/activate

3. Verify the installation:

   .. code:: bash

     cvs --version
     cvs list

If ``cvs --version`` prints a version and ``cvs list`` shows available test suites, CVS is installed correctly.

Install with pip
~~~~~~~~~~~~~~~~

For users who want to install CVS in a custom virtual environment:

1. Clone the repository:

   .. code:: bash

     git clone https://github.com/ROCm/cvs
     cd cvs

2. Build CVS:

   .. code:: bash

     python setup.py sdist

3. Create and activate a Python virtual environment, then install CVS:

   .. code:: bash

     python3 -m venv cvs_env  # or any custom name
     source cvs_env/bin/activate
     pip install dist/cvs*.tar.gz

4. Verify the installation:

   .. code:: bash

     cvs --version
     cvs list

This method gives you more control over the virtual environment name and location.

If ``cvs --version`` prints a version and ``cvs list`` shows available test suites, CVS is installed correctly.

Install and run in a Docker container
-------------------------------------

CVS can run as the head-node CLI in a Docker container. The image connects to
the cluster over SSH; it does not contain ROCm or the workloads that CVS
launches on cluster nodes.

Prerequisites
~~~~~~~~~~~~~

- Docker Engine on the system from which you run CVS.
- Network access from that system to every cluster node over SSH.
- A cluster file and test configuration file prepared as described below.
- An SSH private key that the cluster file references. Do not copy the key into
  the image; mount it read-only at runtime.

Build and verify the image
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, build the image and verify the installed CLI:

.. code:: bash

  docker build --tag cvs:local .
  docker run --rm cvs:local --version
  docker run --rm cvs:local config list-dirs

Two ways to run the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The image sets ``ENTRYPOINT`` to ``cvs``. You can use it in either of these ways:

#. **Long-running container** — override the entrypoint to ``bash`` and keep the container running while you invoke ``cvs`` interactively or with ``docker exec``. Use this when exploring configs or running several commands without restarting the container. See `Long-running container`_.

#. **Run to completion** — pass ``cvs`` subcommands as container arguments (for example ``docker run … cvs:local run …``). The container starts, runs the command, and exits. This is the usual choice for CI and one-shot test runs. See `Run to completion`_.

Long-running container
~~~~~~~~~~~~~~~~~~~~~~

Override the entrypoint to ``bash`` for an interactive shell:

.. code:: bash

  docker run -it --rm --entrypoint bash cvs:local

Inside the container, ``cvs`` is on ``PATH``:

.. code:: bash

  cvs --version
  cvs list

To keep a container running in the background instead of an interactive shell:

.. code:: bash

  docker run -d --name cvs-head --entrypoint bash cvs:local -c "sleep infinity"

Run ``cvs`` via ``docker exec``:

.. code:: bash

  docker exec -it cvs-head cvs --version
  docker exec -it cvs-head cvs list

Run to completion
~~~~~~~~~~~~~~~~~

.. code:: bash

  docker run --rm cvs:local --version
  docker run --rm cvs:local list

The following example runs a test suite. Replace ``<testSuiteName>`` with a name
from ``cvs list``. Prepare a cluster file (``--cluster_file``) and test suite
config (``--config_file``) first—see :doc:`Set up cluster file
</how-to/configure/cluster-config>`, :doc:`Set up test configs
</how-to/configure/test-suite-config/index>`, and :doc:`Run tests </how-to/run-tests/index>`.
Create a host workspace, mount it at ``/workspace`` (read-write) so configs and
run artifacts land on the host, and mount the SSH private key read-only; set
``--config_file`` to the matching JSON under ``/workspace/``:

.. code:: bash

  mkdir -p ~/cvs_workspace

  docker run --rm --init --network host \
    --mount type=bind,src="$HOME/cvs_workspace",dst=/workspace \
    --mount type=bind,src="$HOME/.ssh/id_ed25519",dst=/run/secrets/cvs_ssh_key,readonly \
    cvs:local run <testSuiteName> \
      --cluster_file /workspace/cluster.json \
      --config_file /workspace/<path-to-config.json> \
      --html /workspace/results/<testSuiteName>.html \
      --self-contained-html \
      --log-file /workspace/results/<testSuiteName>.log \
      --capture=tee-sys -vvv -s

``--network host`` gives CVS the same network reachability as the Docker host
on Linux. Remove it only after confirming the bridge network can reach all
cluster nodes. CVS's normal bare-metal execution does not need a Docker socket
inside this image. When the cluster file selects the CVS container backend,
the image still communicates with the remote nodes over SSH; those remote SSH
users need Docker access as documented in :doc:`/how-to/run-with-containers`.


Next steps
==========

- :doc:`/how-to/configure/cluster-config` — configure the cluster file
- :doc:`/how-to/configure/test-suite-config/index` — copy and edit test suite configs
- :doc:`/how-to/run-tests/index` — run tests
