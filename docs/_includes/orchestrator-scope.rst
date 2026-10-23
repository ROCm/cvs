``--cluster_file`` carries ``orchestrator`` (``baremetal`` or ``container``). Write those values in lowercase. The factory accepts any case, and the fixtures that start a container compare against the lowercase string ``container``.

Suites do not all follow the key.

.. list-table::
   :widths: 6 4
   :header-rows: 1

   * - What the suite does
     - Suites
   * - Honors the cluster key. With ``container``, the fixture starts the container and the workload runs inside it (``docker exec``). On more than one node that fixture also starts sshd on port 2224. RCCL host probes (ROCm, RDMA, firewall, dmesg) stay on the host.
     - - ``rvs_cvs``
       - ``install_rvs``
       - ``agfhc_cvs``
       - ``install_agfhc``
       - ``transferbench_cvs``
       - ``install_transferbench``
       - ``csp_qual_agfhc``
       - ``rccl_perf``
       - ``rccl_regression``
       - ``rccl_pairwise``
   * - Honors the key and starts that same container (and sshd on more than one node), then runs its own checks on the host. ``install_ompi`` builds on the host, and its UCX ROCm-path probe also runs on the host using ``orch.head`` or ``orch.all``.
     - - ``preflight_checks``
       - ``host_configs_cvs``
       - ``install_ompi``
   * - Honors the key and skips the suite when the value is ``container``. Baremetal commands run on the host.
     - - ``anc_installation``
       - ``anc_test_*`` suites
   * - Honors the cluster key unless the MORI config sets ``orchestrator``. A test starts the container; the benchmark then runs inside it. sshd is not started. Dmesg stays on the host.
     - - ``mori_benchmark_test``
   * - Builds an orchestrator from the cluster key and never starts a container. Every command is host SSH. A ``container`` value still requires a ``container`` block, and the commands stay on the host.
     - - ``ib_perf_bw_test``
       - ``install_ibperf_tools``
   * - Forces ``container`` and ignores the cluster key. The workload runs inside the container. vLLM does not start sshd. ATOM does. Aorta, SGLang, xDiT, Megatron (``megatron_single`` / ``megatron_distributed``), TorchTitan, and JAX MaxText launch the container and do not call sshd setup.
     - - ``aorta_single``
       - ``aorta_distributed``
       - ``vllm_single``
       - ``vllm_distributed``
       - ``sglang_single``
       - ``sglang_distributed``
       - ``sglang_disagg_distributed``
       - ``xdit_*`` suites
       - ``atom``
       - ``megatron_single``
       - ``megatron_distributed``
       - ``torchtitan_single``
       - ``torchtitan_distributed``
       - ``jaxmaxtext_single``
       - ``jaxmaxtext_distributed``
   * - Ignores ``orchestrator``. The command runs on the host over SSH.
     - - ``cvs exec``
       - ``cvs scp``
