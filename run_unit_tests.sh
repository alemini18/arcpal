#!/usr/bin/env bash

BUILD_DIR="build"
INPUT_DIR="tests/input"
RESULT_DIR="tests/results"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found. Please run 'make' first."
    exit 1
fi

echo "================================================================="
echo " Running Logic Propagator Unit Test Suite"
echo "================================================================="

PASSED=0
FAILED=0
TOTAL=0

for exec in "$BUILD_DIR"/*; do
    [ -f "$exec" ] && [ -x "$exec" ] || continue

    echo "--- $(basename "$exec")"

    for test_file in "$INPUT_DIR"/*.in; do
        [ -e "$test_file" ] || continue

        ((TOTAL++))
        filename=$(basename "$test_file")
        RES_FILE="$RESULT_DIR/${filename}.out"

        OUT=$("$exec" < "$test_file")

        # Dalla prima riga si toglie il numero di iterazioni, che dipende dalla variante
        if diff -q -w <(echo "$OUT" | sed '1s/ [0-9]*$//') "$RES_FILE" > /dev/null; then
            echo -e "[\033[32mPASS\033[0m] $filename"
            ((PASSED++))
        else
            echo -e "[\033[31mFAIL\033[0m] $filename"
            echo "Output:"
            echo "$OUT"
            echo "Expected:"
            cat "$RES_FILE"
            ((FAILED++))
        fi
    done
done

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo "================================================================="

if [ "$TOTAL" -eq 0 ]; then
    echo "Error: No executable found in '$BUILD_DIR'. Please run 'make' first."
    exit 1
fi

[ "$FAILED" -eq 0 ]
