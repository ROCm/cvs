.. meta::
  :description: Identify whether Cluster Validation Suite (CVS) was installed with Makefile, pip, or Docker before you upgrade, uninstall, or downgrade.
  :keywords: CVS, ROCm, install method, Makefile, pip, Docker, which cvs, venv, AMD Instinct, GPU, Linux

*****************************************************************
Identify the Cluster Validation Suite (CVS) install method
*****************************************************************

Upgrade, uninstall, and downgrade must use the same method you used to install
CVS.

``which cvs`` only works if a virtual environment is already active (or
``cvs`` is otherwise on ``PATH``). An empty result often means the venv is not
activated, not that CVS is missing.

From the CVS clone on the head node, use these checks. They do not require an
active venv:

.. list-table::
   :header-rows: 1
   :widths: 5 3

   * - Check
     - Method
   * - ``ls .cvs_venv/bin/cvs`` from the clone succeeds
     - Makefile (``make install``)
   * - ``ls <venv>/bin/cvs`` succeeds for a venv you created
     - pip into a virtual environment
   * - ``docker images cvs`` lists ``cvs:local`` (or the tag you used)
     - Docker image

If a venv is already active, you can also run:

.. code:: bash

  which cvs
  pip show cvs

Typical ``which cvs`` paths when a venv is active:

* ``.../cvs/.cvs_venv/bin/cvs`` — Makefile
* another ``.../bin/cvs`` path — custom pip venv

Next steps
==========

- :doc:`/install/upgrade` — upgrade using that method
- :doc:`/install/uninstall` — uninstall or downgrade using that method
- :doc:`/install/install` — full installation reference
