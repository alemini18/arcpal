#!/usr/bin/env bash

# Define paths
EXEC="$1"
TEST_DIR="tests"

# Ensure the executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable '$EXEC' not found. Please run 'make' first."
    exit 1
fi

# Ensure the data directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo "Error: Directory '$TEST_DIR' not found. Create it and add test files."
    exit 1
fi

echo "================================================================="
echo " Running Logic Propagator Test Suite"
echo "================================================================="

PASSED=0
FAILED=0
TOTAL=0

# Loop through all .in files in the test directory
for test_file in "$TEST_DIR"/*.in; do
    # Safety check in case the directory is empty
    [ -e "$test_file" ] || continue
    
    ((TOTAL++))
    filename=$(basename "$test_file")
    
    # Run the executable, redirecting stdout/stderr to a temporary log
    # Output is hidden unless the test fails
    if $EXEC < $test_file > /tmp/test.log 2>&1; then
        echo -e "[\033[32mPASS\033[0m] $filename"
        ((PASSED++))
    else
        echo -e "[\033[31mFAIL\033[0m] $filename"
        # Print the captured error output indented for easy debugging
        sed 's/^/       /' /tmp/test.log
        ((FAILED++))
    fi
done

# Clean up temp log
rm -f /tmp/test.log

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo "================================================================="

# Exit with an error code if any test failed (useful for CI/CD pipelines)
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi