'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

from cvs.lib.inference.utils.inference_suite_results_table import (
    INFERENCEX_ATOM_RESULTS_COLUMNS,
    make_print_results_table,
)

test_print_results_table = make_print_results_table(INFERENCEX_ATOM_RESULTS_COLUMNS)

__all__ = ["test_print_results_table"]
