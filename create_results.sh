#!/usr/bin/env bash
EXEC=$1

for test_file in tests/sudoku/instance/*.in; do
    [ -e "$test_file" ] || continue

    filename=$(basename "$test_file")

    $EXEC < $test_file > tests/sudoku/results/"$filename".out;

done