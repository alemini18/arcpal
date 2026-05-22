#!/usr/bin/env bash

# Define paths
EXEC="$1"

# Ensure the executable exists
if [ ! -f "$EXEC" ]; then
    echo "Error: Executable '$EXEC' not found. Please run 'make' first."
    exit 1
fi

echo "================================================================="
echo " Running Logic Propagator Test Suite"
echo "================================================================="

PASSED=0
FAILED=0
TOTAL=0

python3 compact_to_dimacs.py $2

# Loop through all .in files in the test directory
for test_file in tests/sudoku/input/*.in; do
    # Safety check in case the directory is empty
    [ -e "$test_file" ] || continue
    
    ((TOTAL++))
    filename=$(basename "$test_file")
    
    # Run the executable, redirecting stdout/stderr to a temporary log
    # Output is hidden unless the test fails
    $EXEC < $test_file > tests/sudoku/output/"$filename".tmp 2>&1;
    python3 dimacs_to_compact.py tests/sudoku/output/"$filename".tmp > tests/sudoku/output/"$filename".out
    if diff -q -w tests/sudoku/output/"$filename".out tests/sudoku/results/"$filename".res; then
        echo -e "[\033[32mPASS\033[0m] $filename"
        ((PASSED++))
    else
        echo -e "[\033[31mFAIL\033[0m] $filename"
        # Print the captured error output indented for easy debugging
        echo "Output:"
        cat tests/sudoku/output/"$filename".out 
        echo "Expected:"
        cat tests/sudoku/results/"$filename".res
        ((FAILED++))
    fi
done

# Clean up temp log
rm -f tests/sudoku/input/*
rm -f tests/sudoku/output/*
rm -r tests/sudoku/results/*

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo "================================================================="

# Exit with an error code if any test failed (useful for CI/CD pipelines)
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi