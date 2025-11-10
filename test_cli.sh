#!/bin/bash
# Test script for CVS package
# This script tests the cvs list and cvs generate commands

set -e

# Use the CVS environment variable if set, otherwise default to 'cvs'
CVS="${CVS:-cvs}"

echo "Testing cvs commands..."
echo "===================="
echo ""

echo "Testing: cvs list (list all tests)"
output="$($CVS list 2>&1)"
echo "$output"
if echo "$output" | grep -qi "ERRORS ===\|ERROR collecting\|Traceback\|!!!.*error"; then
    echo ""
    echo "FAILED: cvs list - errors detected in output"
    exit 1
fi

echo ""
echo "Testing: cvs list <test_name> for each test suite"
echo "===================="
set -e
for test in $($CVS list | grep -v "Available tests:" | grep -v "^$$" | awk '{print $2}'); do
    echo ""
    echo "Testing: cvs list $test"
    output="$($CVS list $test 2>&1)"
    echo "$output"
    if echo "$output" | grep -qi "ERRORS ===\|ERROR collecting\|Traceback\|!!!.*error"; then
        echo ""
        echo "FAILED: cvs list $test - errors detected in output"
        exit 1
    fi
done

echo ""
echo "===================="
echo "All cvs list command tests passed!"
echo ""

echo "Testing: cvs generate (list all generators)"
echo "===================="
output="$($CVS generate 2>&1)"
echo "$output"
if echo "$output" | grep -qi "Error:\|Traceback\|No module named"; then
    echo ""
    echo "FAILED: cvs generate - errors detected in output"
    exit 1
fi

echo ""
echo "Testing: cvs generate <generator> help for each generator"
echo "===================="
set -e
for generator in $($CVS generate | grep -v "Available generators:" | grep -v "^$$" | awk '{print $1}'); do
    echo ""
    echo "Testing: cvs generate $generator -h"
    output="$($CVS generate $generator -h 2>&1)"
    echo "$output"
    if echo "$output" | grep -qi "Error:\|Traceback\|No module named"; then
        echo ""
        echo "FAILED: cvs generate $generator -h - errors detected in output"
        exit 1
    fi
done

echo ""
echo "===================="
echo "All cvs generate command tests passed!"
echo ""

echo "Testing: cvs run (list all tests)"
echo "===================="
output="$($CVS run 2>&1)"
echo "$output"
if echo "$output" | grep -qi "ERRORS ===\|ERROR collecting\|Traceback\|!!!.*error"; then
    echo ""
    echo "FAILED: cvs run - errors detected in output"
    exit 1
fi

echo ""
echo "Testing: cvs run <test_name> --collect-only for each test suite"
echo "===================="
set -e
for test in $($CVS run | grep -v "Available tests:" | grep -v "^$$" | awk '{print $2}'); do
    echo ""
    echo "Testing: cvs run $test --collect-only"
    output="$($CVS run $test --collect-only 2>&1)"
    echo "$output"
    if echo "$output" | grep -qi "ERRORS ===\|ERROR collecting\|Traceback\|!!!.*error"; then
        echo ""
        echo "FAILED: cvs run $test --collect-only - errors detected in output"
        exit 1
    fi
done

echo ""
echo "===================="
echo "All cvs run command tests passed!"