.. meta::
  :description: Create and update a rack-aware CVS cluster file with cvs generate rack_cluster_json, including adding racks and switch or BMC details later with --amend and --replace.
  :keywords: CVS, rack cluster file, cluster_rack.json, racks, rack_cluster_json, amend, replace, ARC, switch trays, BMC, RMC

********************************************************
Cluster Validation Suite (CVS) rack-aware cluster file
********************************************************

A rack-aware cluster file is a :doc:`cluster file <cluster-file>` with two additions: each compute node in ``node_dict`` has a ``rack_id``, and a ``racks`` block describes each rack's devices and the logins for them. Suites that do not use racks ignore both, so one rack-aware file serves every suite.

When you need one
=================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Use
     - What it reads from the ``racks`` block
   * - ARC suites (``arc_installation``, ``arc_test_*``)
     - ``platform`` and the device lists, to generate each rack's ARC rack file and pick its controller. See :doc:`/how-to/test-suites/burn-in-diag/arc`.
   * - ``cvs exec --target switches`` or ``--target all``
     - ``switch_trays`` and the ``switch_ssh_*`` login. See :doc:`/how-to/execute-cluster-commands`.

Every other suite needs only a plain cluster file.

What goes in it
===============

- **Compute nodes**, tagged with their ``rack_id``. List each rack's nodes **in tray order**. On Helios, the compute tray in U9 comes first and the one in U33 last. ARC derives tray numbers from this order.
- **A** ``platform`` **per rack**: ``HeliosP``, ``HeliosR``, or ``HeliosM``.
- **Optional devices**: switch trays, switch BMCs, compute BMCs, rack management controllers (RMC), front-end and back-end switches, and power shelves. List only what you can reach. Without switch or BMC access, the compute-only file is complete; set ``arc.spec_mode`` to ``partial`` for ARC.
- **Optional logins**: one switch login and one BMC login for the whole file.

For what each field means and the slot order of every device type, see :ref:`Rack inventory in the cluster file <arc-rack-inventory>`.

Create the file
===============

Copy the sample and edit it, or generate the file from address ranges. Generating is quicker for full racks, because ranges such as ``10.0.0.1-18`` or ``node[01-18]`` expand in order.

Copy the sample:

.. code:: bash

   cvs config copy cluster_rack.json --output ~/cluster_rack.json

Generate a file with one compute-only rack:

.. code:: bash

   cvs generate rack_cluster_json --username <user> --key_file ~/.ssh/id_rsa \
     --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-01 --platform HeliosP --computes 10.0.0.1-18

Generate a file with two racks, one with switch, BMC, and RMC details:

.. code:: bash

   cvs generate rack_cluster_json --username <user> --key_file ~/.ssh/id_rsa \
     --switch_ssh_user <switch_user> --bmc_user <bmc_user> \
     --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-01 --platform HeliosP --computes 10.0.0.1-18 --bmcs 10.0.1.1-18 \
             --switches 10.0.2.1-6 --switch_bmc 10.0.3.1-6 --rmc 10.0.4.1 \
     --rack1 --id rack-02 --platform HeliosR --computes 10.1.0.1-18

Each ``--rackN`` flag starts one rack, and the flags after it belong to that rack. Put the flags that apply to the whole file, such as ``--username``, ``--bmc_user``, and ``--output_json_file``, before ``--rack0``.

Pass ``--bmc_password`` only if you accept it in your shell history. Without it, the file holds ``<changeme>``, and ARC suites stop until you replace it. Generating without ``--amend`` overwrites ``--output_json_file``.

Update the file
===============

Add ``--amend`` to change an existing file instead of overwriting it. With ``--amend``, every ``--rackN`` group needs ``--id``, and ``--username`` and ``--key_file`` are optional because they come from the file. Everything not named in the command stays as it is, including fields you edited by hand.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - To
     - Run
     - Result
   * - Add a rack
     - ``--amend`` with a new ``--id`` and its ``--computes``
     - The rack and its nodes are added after the existing ones.
   * - Add details to a rack, such as switches when you gain access
     - ``--amend`` with that rack's ``--id`` and only the new flags
     - Fields the rack lacks are filled in. ``--computes`` is optional.
   * - Correct a rack's details
     - ``--amend --replace`` with that rack's ``--id`` and the corrected flags
     - The fields you pass overwrite the file's. The rest of the rack is kept.
   * - Change a rack's compute nodes
     - ``--amend --replace`` with that rack's ``--id`` and the new ``--computes``
     - The rack's nodes are replaced in place. A node that stays keeps its existing entry.

Add a second rack:

.. code:: bash

   cvs generate rack_cluster_json --amend --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-02 --platform HeliosP --computes 10.1.0.1-18

Add switch details to a rack that started with computes only:

.. code:: bash

   cvs generate rack_cluster_json --amend --switch_ssh_user <switch_user> \
     --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-01 --switches 10.0.2.1-6

Add BMC details later:

.. code:: bash

   cvs generate rack_cluster_json --amend --bmc_user <bmc_user> \
     --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-01 --bmcs 10.0.1.1-18 --switch_bmc 10.0.3.1-6 --rmc 10.0.4.1

Correct a wrong switch list:

.. code:: bash

   cvs generate rack_cluster_json --amend --replace --output_json_file ~/cluster_rack.json \
     --rack0 --id rack-01 --switches 10.0.2.11-16

What stops an update
--------------------

``--amend`` never overwrites a value silently. These stop the command and leave the file unchanged:

- A rack field that already holds a different value, without ``--replace``. Passing the same value again is allowed.
- A login (``--username``, ``--key_file``, ``--switch_ssh_*``, ``--bmc_user``, ``--bmc_password``) or ``--head_node`` that differs from the file's. ``--replace`` does not change logins; edit the file instead. A login that is missing or still ``<changeme>`` is filled in.
- ``--computes`` for an existing rack that differ from the file's list, without ``--replace``.
- A compute node that already belongs to another rack.
- ``--bmcs`` whose count differs from the rack's compute nodes, or ``--switch_bmc`` whose count differs from its switch trays, counting entries already in the file.
- An ``--arc_controller`` that is not one of the rack's compute nodes.

An updated file keeps its permissions. If it holds passwords, restrict it with ``chmod 600``.

Flags and the fields they set
=============================

Run ``cvs generate rack_cluster_json --help`` for the full list.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Field
   * - ``--username``, ``--key_file``
     - ``username``, ``priv_key_file``
   * - ``--head_node``
     - ``head_node_dict.mgmt_ip``. Default: the first compute node.
   * - ``--switch_ssh_user``, ``--switch_ssh_password``, ``--switch_ssh_key_file``
     - ``racks.switch_ssh_user``, ``racks.switch_ssh_password``, ``racks.switch_ssh_key_file``
   * - ``--bmc_user``, ``--bmc_password``
     - ``racks.bmc_user``, ``racks.bmc_password``
   * - ``--id``
     - The rack's key under ``racks`` and each of its nodes' ``rack_id``
   * - ``--computes``, ``--bmcs``
     - ``node_dict`` entries in the order given, with ``vpc_ip`` set to the address and ``bmc_ip`` from ``--bmcs`` (``NA`` without it)
   * - ``--platform``
     - ``racks.<id>.platform``. Default for a new rack: ``HeliosP``.
   * - ``--arc_controller``, ``--switch_model``
     - ``racks.<id>.arc_controller``, ``racks.<id>.switch_model``
   * - ``--switches``, ``--switch_bmc``
     - ``racks.<id>.switch_trays``, ``racks.<id>.switch_bmc``
   * - ``--rmc``, ``--power_shelves``
     - ``racks.<id>.rmc``, ``racks.<id>.power_shelves``
   * - ``--front_end_switches``, ``--back_end_switches``
     - ``racks.<id>.front_end_switches``, ``racks.<id>.back_end_switches``
