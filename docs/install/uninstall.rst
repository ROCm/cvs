.. meta::
  :description: Uninstall or downgrade Cluster Validation Suite (CVS) on the head node, and identify whether you installed with Makefile, pip, or Docker.
  :keywords: CVS, ROCm, uninstall, downgrade, pip, Makefile, Docker, AMD Instinct, GPU, Linux

*****************************************************************
Uninstall or downgrade Cluster Validation Suite (CVS) on ROCm
*****************************************************************

These steps remove or roll back the CVS *CLI on the head node*. They don't
uninstall ROCm, drivers, or packages on cluster workers. 

Identify how CVS was installed
==============================

.. include:: /_includes/identify.rst

Uninstall CVS
=============

If installed with ``make install``
----------------------------------

Deactivate the environment, then remove the venv that ``make install`` created:

.. code:: bash

  deactivate
  cd /path/to/cvs/source
  rm -rf .cvs_venv
  which cvs

Expected output after a successful removal:

.. code:: text

  $ which cvs
  $

``cvs`` is no longer on ``PATH``. Delete the clone if you no longer need the
source tree.

If installed with pip in a custom venv
--------------------------------------

.. code:: bash

  pip uninstall cvs
  which cvs

Expected output:

.. code:: text

  Found existing installation: cvs 0.2.0
  Uninstalling cvs-0.2.0:
    Successfully uninstalled cvs-0.2.0

  $ which cvs
  $

Then ``deactivate`` and remove the venv directory if you no longer need it.

If installed as a Docker image
------------------------------

Stop a named container if you created one, then remove the image:

.. code:: bash

  docker rm -f cvs-head
  docker rmi cvs:local
  docker images cvs

Expected output:

.. code:: text

  Untagged: cvs:local
  Deleted: sha256:...

  $ docker images cvs
  REPOSITORY   TAG   IMAGE ID   CREATED   SIZE

Downgrade CVS
=============

CVS is installed from source. To move to an older release, check out that
version and reinstall with the *same* method you originally used. 

Example using ``make install``:

.. code:: bash

  cvs --version
  cd /path/to/cvs/source
  git fetch origin
  git checkout <older-version>
  make install
  source .cvs_venv/bin/activate
  cvs --version

Expected output:

.. code:: text

  $ cvs --version
  0.2.0
  ...
  $ cvs --version
  0.1.0

Use ``pip install dist/cvs*.tar.gz`` or ``docker build --tag cvs:local .``
instead if that is how you originally installed CVS.

Next steps
==========

- :doc:`/install/install` — reinstall CVS
- :doc:`/install/upgrade` — upgrade to the latest source
- :doc:`/install/quickstart` — install and run the first cluster command
