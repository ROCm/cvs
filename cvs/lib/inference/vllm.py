'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import time
from pprint import pprint
from tabulate import tabulate

from cvs.lib import globals
from cvs.lib.inference.base import InferenceBaseJob
from cvs.lib.utils_lib import update_test_result


class VllmJob(InferenceBaseJob):
    """vLLM-specific implementation."""

    # Class-level results dictionary shared across all test iterations
    all_test_results = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_dict.setdefault('benchmark_server_script_path', '/host_scripts')

    def get_server_script_directory(self):
        """vLLM scripts are mounted from host."""
        return self.if_dict['benchmark_server_script_path']

    def get_result_filename(self):
        """vLLM result filename."""
        return 'vllm_test_result.json'

    def get_completion_pattern(self):
        """vLLM completion pattern."""
        return re.compile('End-to-end Latency', re.I)

    def get_log_subdir(self):
        """vLLM uses 'vllm' log subdirectory."""
        return 'vllm'

    def stop_server(self):
        """Stop the vLLM server process."""
        log = globals.log
        log.info("Stopping vLLM server")
        self.s_phdl.exec(f'docker exec {self.container_name} pkill -f "vllm serve"')
        time.sleep(5)  # Wait for graceful shutdown

    def restart_server(self):
        """Restart the vLLM server with updated parameters."""
        log = globals.log
        log.info("Restarting vLLM server with updated parameters")
        self.stop_server()
        self.build_server_inference_job_cmd()
        self.start_inference_server_job()

    def store_test_result(self):
        """
        Store test results from the last poll_for_inference_completion call.

        Automatically extracts test parameters from the instance's benchmark_params_dict.
        """
        # Get the last poll result (assumes poll_for_inference_completion was just called)
        if hasattr(self, 'inference_result_dict') and self.inference_result_dict:
            # Extract test parameters from the instance
            isl = self.bp_dict['input_sequence_length']
            osl = self.bp_dict['output_sequence_length']
            conc = int(self.bp_dict['max_concurrency'])

            # Find the sequence combination name from config
            seq_name = "unknown"
            seq_combos = self.bp_dict.get("sequence_combinations", [])
            for combo in seq_combos:
                if combo['isl'] == isl and combo['osl'] == osl:
                    seq_name = combo.get('name', f"isl{isl}_osl{osl}")
                    break

            res_index = (self.model_name, self.gpu_type, isl, osl, seq_name, conc)
            # Store with the same structure as poll_for_inference_completion returns
            VllmJob.all_test_results[res_index] = {"status": "success", "results": self.inference_result_dict}

    @classmethod
    def print_all_results(cls):
        """
        Print a formatted table of all accumulated test results.

        Displays a GitHub-formatted markdown table showing all test results including:
            - Model, GPU, ISL, OSL, Policy, Concurrency
            - Per-node metrics: Req/s, Total tok/s, Mean TTFT, Mean TPOT, P99 ITL
        """
        globals.error_list = []
        print(cls.all_test_results)
        pprint(cls.all_test_results, depth=3)

        rows = []
        headers = [
            "Model",
            "GPU",
            "ISL",
            "OSL",
            "Policy",
            "Conc",
            "Host",
            "Req/s",
            "Tot tok/s",
            "Mean TTFT (ms)",
            "Mean TPOT (ms)",
            "P99 ITL (ms)",
        ]

        for (model, gpu, isl, osl, policy, conc), entry in cls.all_test_results.items():
            for host, m in entry["results"].items():
                rows.append(
                    [
                        model,
                        gpu,
                        isl,
                        osl,
                        policy,
                        conc,
                        host,
                        m["successful_requests"],
                        m["total_throughput_per_sec"],
                        m["mean_ttft_ms"],
                        m["mean_tpot_ms"],
                        m["p99_itl_ms"],
                    ]
                )

        print(tabulate(rows, headers=headers, tablefmt="github"))
        update_test_result()

    @classmethod
    def clear_all_results(cls):
        """Clear all accumulated test results. Useful for test isolation."""
        cls.all_test_results = {}


def print_inference_results_table(inf_res_dict):
    """
    DEPRECATED: Use VllmJob.print_all_results() instead.

    Print a formatted table of inference test results.

    Args:
        inf_res_dict: Dictionary with structure:
            {(model, gpu, isl, osl, policy, conc): {"status": ..., "results": {node: metrics}}}
    """
    globals.error_list = []
    print(inf_res_dict)
    pprint(inf_res_dict, depth=3)

    rows = []
    headers = [
        "Model",
        "GPU",
        "ISL",
        "OSL",
        "Policy",
        "Conc",
        "Host",
        "Req/s",
        "Tot tok/s",
        "Mean TTFT (ms)",
        "Mean TPOT (ms)",
        "P99 ITL (ms)",
    ]

    for (model, gpu, isl, osl, policy, conc), entry in inf_res_dict.items():
        for host, m in entry["results"].items():
            rows.append(
                [
                    model,
                    gpu,
                    isl,
                    osl,
                    policy,
                    conc,
                    host,
                    m["successful_requests"],
                    m["total_throughput_per_sec"],
                    m["mean_ttft_ms"],
                    m["mean_tpot_ms"],
                    m["p99_itl_ms"],
                ]
            )

    print(tabulate(rows, headers=headers, tablefmt="github"))
    update_test_result()
