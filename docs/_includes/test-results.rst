.. tip::

  If errors prevent tests from starting, CVS prints a message with a suggested fix in the console.

.. image:: /images/install.png

Install suites run the tool installation commands on each node and confirm the
binary is on the expected path.

.. image:: /images/success.png

All tests passed. The console shows per-step timing and ends with the pytest
summary line and an HTML report path.

.. image:: /images/failed.png

One or more tests failed. The short test summary lists each failing test with
the parametrization that triggered it and the assertion that failed.

.. image:: /images/stop.png

The run was interrupted with **Ctrl** + **C**. CVS prints the ``KeyboardInterrupt``
traceback and generates a partial HTML report for the tests that completed.
