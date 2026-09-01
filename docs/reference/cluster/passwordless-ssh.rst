.. meta::
  :description: Enable passwordless SSH for CVS cluster nodes
  :keywords: SSH, cluster, passwordless, authorized_keys

*******************
Passwordless SSH
*******************

CVS requires SSH key-based access from the head node to every cluster worker. Passwordless login must work for the user named in ``cluster.json`` (``username`` and ``priv_key_file``).

.. tip::

  Perform these steps in reverse order (child node first, then head node) if you require passwordless login from a head node to a child node.

1. On the head node, display your public key:

   .. code:: bash

     cat ~/.ssh/id_rsa.pub

2. On each worker node, add that key to ``authorized_keys``:

   .. code:: bash

     echo "paste-your-public-key-here" >> ~/.ssh/authorized_keys
     chmod 600 ~/.ssh/authorized_keys

3. Verify login from the head node:

   .. code:: bash

     ssh username@remote_host_ip

4. If the username is the same on both nodes, you can use the IP only:

   .. code:: bash

     ssh remote_host_ip

You can use any SSH key type, not only RSA. Point ``priv_key_file`` in :doc:`/reference/cluster/cluster-file` at the matching private key.

See also :doc:`/how-to/configure/cluster-config` for generating and editing ``cluster.json``.
