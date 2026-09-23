.. meta::
  :description: Upgrade CVS to the latest version on ROCm AMD Instinct GPU clusters using Makefile or pip from source on Linux.
  :keywords: CVS, ROCm, upgrade, install, AMD Instinct, GPU, AMD, pip, Makefile, Linux, cluster, update

**********************************************************
Upgrade Cluster Validation Suite (CVS) to the latest version
**********************************************************

Upgrade CVS after pulling the latest source from the repository. Follow the
method that matches how you originally installed CVS. 

Identify how CVS was installed
==============================

.. include:: /_includes/identify.rst

CVS upgrade paths
=================

If installed with ``make install``
----------------------------------

.. code:: bash

  cd /path/to/cvs/source
  git pull
  make install
  source .cvs_venv/bin/activate

If installed manually in a custom venv
--------------------------------------

.. code:: bash

  cd /path/to/cvs/source
  git pull
  python setup.py sdist
  pip install --upgrade dist/cvs*.tar.gz

If installed as a Docker image
------------------------------

Rebuild the image from the updated source:

.. code:: bash

  cd /path/to/cvs/source
  git pull
  docker build --tag cvs:local .
  docker run --rm cvs:local --version

The new image replaces the old one locally. If you use a named container (``cvs-head``), stop and remove it before rebuilding so the new image is used on the next ``docker run``.

Verify
======

Run the following commands to verify the upgrade:

.. code:: bash

  cvs --version
  cvs list

If ``cvs --version`` prints the updated version number and ``cvs list`` shows available test suites, the upgrade was successful.

Next steps
==========

- :doc:`/how-to/test-suites/index` — run tests against the cluster
- :doc:`/install/install` — full installation reference for all methods
- :doc:`/install/uninstall` — uninstall or downgrade CVS
