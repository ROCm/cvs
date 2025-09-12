'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import re
import os
import sys
import json

import pytest
import globals

log = globals.log

def fail_test(msg):
    """
    Record and report a test failure without immediately raising an exception.
    This will enable Pytest to continue running further steps in the test case 
    without returning back on first failure.

    Parameters:
      msg (str): Human-readable failure description to log and store.

    Behavior:
      - Prints a standardized "FAIL - ..." message to stdout for quick visibility.
      - Logs the same message at error level via the global `log` logger.
      - Appends the raw message to `globals.error_list` for later aggregation/reporting.
      - Does NOT raise/abort by default. The pytest failure line is present but commented out.

    Notes:
      - If you want to stop execution on first failure (e.g., in pytest),
        uncomment the pytest.fail(...) line.
      - Assumes:
          * `log` is a configured logger available in module scope.
          * `globals.error_list` is an initialized list collecting error messages.
    """
    print('FAIL - {}'.format(msg))
    log.error('FAIL - {}'.format(msg))
    # We append the error messages to the global error list which will be checked at the
    # end of every test case in Pytest to determine test PASS/FAIL
    globals.error_list.append(msg)



def update_test_result():
    # For every Pytest test case, we will initialize the global.error_list to empty and
    # whenever fail_test is executed, it would add the error messages and if this is 
    # a non-zero list, then the test case will be marked as a failure in the end of the
    # test case
    if len(globals.error_list) > 0:
        pytest.fail('Following FAILURES seen - {}'.format(globals.error_list))



def print_test_output(log, out_dict):
    print('#========================================================#')
    print('\t\t ** Test Output **')
    print('#========================================================#')
    for node in out_dict.keys():
        print(f'==== {node} ====')
        print(out_dict[node]) 



def scan_test_results(out_dict):
    """
    Scan test outputs from multiple hosts for failure indicators and report context.

    Parameters:
      out_dict (dict): Mapping of host -> output string (e.g., test logs or stdout/stderr).

    Behavior:
      - For each host, searches the output for common failure patterns:
          'test FAIL', 'test ERROR', 'ABORT', 'Traceback', 'No such file', 'FATAL' (case-insensitive).
      - When a match is found, splits the full output into words and locates the first token
        containing any of: 'FAIL', 'ERR', 'ABORT', 'Traceback'.
      - Prints up to 5 words before and after the matched token to provide context.
      - Calls fail_test with a message that includes the before/actual/after tokens.

    Notes:
      - Assumes `re` is imported, and `fail_test` is available in scope.
      - word_count controls the number of contextual words (currently set to 5).
      - Potential issue: `fail_test` is called unconditionally at the end of the loop iteration,
        which will raise even when no match is found, and before_words/actual_word/after_words
        might be undefined if no target word was located. Consider moving the fail_test call
        inside the `if target_word_index != -1:` block and guarding for no match cases.
    """

    word_count = 5  # Number of words to include before/after the matched token for context
   
    # Iterate over each host's output 
    for host in out_dict.keys():
        # Search for any high-level failure pattern in the raw output
        match = re.search( 'test FAIL |test ERROR |ABORT|Traceback|No such file|FATAL', out_dict[host], re.I )
        if match:
            # Record the span of the first match (currently unused; could help with slicing)
            start_index = match.start()
            end_index = match.end()
            # Tokenize the entire output into words for contextual extraction
            words = out_dict[host].split()
            # Find the index of the target word in the list of words
            target_word_index = -1
            for i, word in enumerate(words):
                # Check if this token contains any of the target substrings
                if re.search('FAIL|ERR|ABORT|Traceback', word, re.I): # Check if the pattern is part of the word
                    target_word_index = i
                    actual_word = word
                    break

            if target_word_index != -1:
                # Collect up to 'word_count' words immediately before the matched token
                # Get the words before the match
                before_words = words[max(0, target_word_index - word_count) : target_word_index]
                # Get the words after the match
                after_words = words[target_word_index + 1 : min(len(words), target_word_index + word_count + 1)]
                print(f"Words before: {before_words}")
                print(f"Words after: {after_words}")
                print(f'Test failed in scan_result on node {host} due to pattern ')
            fail_test(f'Test failed in scan_result on node {host} due to pattern {before_words} {actual_word} {after_words}')




def json_to_dict(json_string):
    print('^^^^^^^^^')
    print(json_string)
    return json.loads(json_string) 


def convert_phdl_json_to_dict( dict_json ):

    """
    Convert a per-node JSON payload map into a per-node Python dict map.

    Args:
      dict_json (dict): Mapping of node identifier -> JSON string (or JSON-like object)
                        that can be parsed by json_to_dict.

    Returns:
      dict: Mapping of node identifier -> parsed Python dictionary.
            If parsing fails for a node, its value will be an empty dict and a failure
            will be recorded via fail_test.

    Behavior:
      - Iterates over each node key in dict_json.
      - Attempts to parse the node's JSON payload using json_to_dict.
      - On parse error, prints/logs a failure message, calls fail_test, and assigns {} for that node.

    Assumptions:
      - json_to_dict is available in scope and raises an Exception on invalid JSON.
      - fail_test is available in scope for reporting failures.
      - dict_json keys are node identifiers; values are JSON strings or compatible objects.
    """

    out_dict = {}   # Will hold node -> parsed dict

    # Process each node's JSON blob and convert to a Python dict
    for node in dict_json.keys():
        try:
            # Parse the JSON content for this node
            out_dict[node] = json_to_dict( dict_json[node] )
        except Exception as e:
            print(f'ERROR converting Json output to dict for node {node}')
            fail_test(f'ERROR converting Json output to dict for node {node}')
            out_dict[node] = {}
    return out_dict



def get_model_from_rocm_smi_output(smi_output):
    """
    Infer the GPU model identifier from a rocm-smi output snippet.

    Args:
      smi_output (str): Text output captured from `rocm-smi -a` or similar.

    Returns:
      str: A normalized model string, one of:
           - 'mi300x' (default if no known model is matched)
           - 'mi325'
           - 'mi350'
           - 'mi355'

    Behavior:
      - Performs case-insensitive regex searches for specific model tokens in the provided
        rocm-smi output (e.g., "MI300X", "MI325", "MI350", "MI355").
      - Returns the corresponding normalized lowercase token (e.g., 'mi300x').
      - Falls back to 'mi300x' if no pattern matches (conservative default).

    """
    if re.search( 'MI300X', smi_output, re.I ):
        model = 'mi300x'  
    elif re.search( 'MI325', smi_output, re.I ):
        model = 'mi325' 
    elif re.search( 'MI350', smi_output, re.I ):
        model = 'mi350'
    elif re.search( 'MI355', smi_output, re.I ):
        model = 'mi355'
    else:  
        model = 'mi300x'
    return model



def convert_hms_to_secs( time_string ):
    """
    Convert a time string in 'HH:MM:SS' format to total seconds.

    Args:
      time_string (str): Time formatted as 'hours:minutes:seconds'.
                         - Hours and minutes should be integers.
                         - Seconds may be integer or float (e.g., '12.5').

    Returns:
      float | None:
        - Total seconds as a float (to support fractional seconds) when parsing succeeds.
        - None if the input format is invalid or contains non-numeric components.

    Behavior:
      - Splits the input on ':' and expects exactly 3 parts.
      - Parses hours and minutes as integers, seconds as float.
      - Computes total_seconds = hours*3600 + minutes*60 + seconds.
      - Prints a user-friendly error message and returns None on format/parse errors.

    """ 
    try:
        # Break the time string into components by ':'
        parts = time_string.split(':')

        # Expect exactly three components: HH:MM:SS
        if len(parts) == 3:
            # Convert hours and minutes to int; seconds to float to allow decimal seconds
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])  # Use float to handle potential decimal seconds
            # Compute total seconds, preserving fractional seconds
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
            return total_seconds
        else:
            print("Invalid time format. Please use 'hr:min:sec'.")
            return None
    except ValueError:
        print("Invalid time format. Please ensure hours, minutes, and seconds are numeric.")
        return None
