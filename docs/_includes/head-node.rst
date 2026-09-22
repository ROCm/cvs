.. note::

  **Head node.** The Linux host where you install and run the CVS CLI. It can be
  a VM or bare metal and does not need a GPU. It must be able to SSH to every
  worker.

  The head node can be either of the following:

  * One of the cluster members in ``node_dict`` — usually the first node.
  * A completely separate host that is not in ``node_dict``.

  Set ``head_node_dict.mgmt_ip`` in ``cluster.json`` to that host's address.
  See :doc:`/reference/cluster/cluster-file`.
