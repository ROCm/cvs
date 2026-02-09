#!/bin/bash
# Test script for CVS package
# This script tests the cvs list, generate, copy-config, monitor, and exec commands

# Use the CVS environment variable if set, otherwise default to 'cvs'
CVS="${CVS:-cvs}"

echo "Testing cvs commands..."
echo "===================="
echo ""

FAILED_TESTS=()
PASSED_TESTS=()
PIDS=()
TEST_NAMES=()
TEST_COMMANDS=()

# Function to run a test in background
run_test() {
    local test_name="$1"
    local command="$2"

    echo "Testing: $test_name"
    echo "Command: $command"

    # Run command in background and capture PID
    eval "$command" > /tmp/test_output_$$_$BASHPID 2>&1 &
    local pid=$!
    PIDS+=($pid)
    TEST_NAMES+=("$test_name")
    TEST_COMMANDS+=("$command")
    echo "Started test (PID: $pid)"
    echo ""
}

# Function to collect results from completed tests
collect_results() {
    local num_tests=${#PIDS[@]}
    for ((i=0; i<num_tests; i++)); do
        local pid=${PIDS[$i]}
        local test_name="${TEST_NAMES[$i]}"
        local command="${TEST_COMMANDS[$i]}"
        local output_file="/tmp/test_output_$$_$pid"

        # Wait for this specific process
        if wait $pid 2>/dev/null; then
            echo "PASSED: $test_name"
            PASSED_TESTS+=("$test_name")
        else
            echo "FAILED: $test_name"
            echo "Command was: $command"
            if [ -f "$output_file" ]; then
                echo "Output:"
                cat "$output_file"
            fi
            FAILED_TESTS+=("$test_name")
        fi

        # Clean up output file
        rm -f "$output_file"
        echo ""
    done
}

# Test: cvs list (list all tests)
run_test "cvs list" "$CVS list"

# Test: cvs list <test_name> for each test suite
echo "Testing: cvs list <test_name> for each test suite"
echo "===================="
for test in $($CVS list | grep "•" | awk '{print $2}'); do
    run_test "cvs list $test" "$CVS list $test"
done

# Test: cvs generate (list all generators)
run_test "cvs generate" "$CVS generate"

# Test: cvs generate <generator> help for each generator
echo "Testing: cvs generate <generator> help for each generator"
echo "===================="
for generator in $($CVS generate | grep -v "Available generators:" | grep -v "^$" | awk '{print $1}'); do
    run_test "cvs generate $generator -h" "$CVS generate $generator -h"
done

# Test: cvs copy-config --list
run_test "cvs copy-config --list" "$CVS copy-config --list"

# Test: cvs monitor (list all monitors)
run_test "cvs monitor" "$CVS monitor"

# Test: cvs monitor <monitor> help for each monitor
echo "Testing: cvs monitor <monitor> help for each monitor"
echo "===================="
for monitor in $($CVS monitor | grep -v "Available monitors:" | grep -v "^$" | awk '{print $1}'); do
    run_test "cvs monitor $monitor -h" "$CVS monitor $monitor -h"
done

# Test: cvs exec --help
run_test "cvs exec --help" "$CVS exec --help"

# Wait for all tests to complete and collect results
echo "Waiting for all tests to complete..."
echo "===================="
collect_results

# Summary
echo "===================="
echo "Test Summary:"
echo "===================="
echo "Passed: ${#PASSED_TESTS[@]}"
for test in "${PASSED_TESTS[@]}"; do
    echo "  ✓ $test"
done

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED_TESTS[@]}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
    echo "Some tests failed. Please check the output above for details."
    exit 1
else
    echo ""
    echo "All CVS CLI tests passed successfully!"
    echo "===================="
fi
echo ""