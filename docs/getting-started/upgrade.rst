.. meta::
  :description: Upgrade Cluster Validation Suite to the latest version
  :keywords: CVS, upgrade, install

*******
Upgrade
*******

Upgrade CVS after pulling the latest source from the repository.

If installed with ``make install``
==================================

.. code:: bash

  cd /path/to/cvs/source
  git pull
  make install
  source .cvs_venv/bin/activate

If installed manually in a custom venv
======================================

.. code:: bash

  cd /path/to/cvs/source
  git pull
  python setup.py sdist
  pip install --upgrade dist/cvs*.tar.gz

Verify
======

.. code:: bash

  cvs --version
  cvs list

See also :doc:`/getting-started/install`.
