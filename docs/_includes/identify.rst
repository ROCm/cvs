
To identify how CVS was installed, use these commands from the cloned CVS on the head node (they don't require an
active venv):

.. list-table::
   :header-rows: 1
   :widths: 3 5

   * - Command
     - Installation via
   * - ``ls .cvs_venv/bin/cvs`` 
     - Makefile (``make install``) if the command is succesful
   * - ``ls <venv>/bin/cvs`` 
     - pip into a virtual environment if the command succeeds for a venv you created
   * - ``docker images cvs`` 
     - Docker image if the command lists ``cvs:local`` (or the tag you used)

If a venv is already active, you can also run:

.. code:: bash

  which cvs
  pip show cvs

Typical ``which cvs`` paths when a venv is active:

* ``.../cvs/.cvs_venv/bin/cvs`` — Makefile
* Another ``.../bin/cvs`` path — custom pip venv

.. note::

   ``which cvs`` only works if a virtual environment is already active (or ``cvs`` is otherwise on ``PATH``). An empty result often means the venv is not activated, not that CVS is missing.