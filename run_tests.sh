#!/usr/bin/env bash

EXEC="$1"

exec_name="$(basename "$EXEC")"

if [ ! -f "$EXEC" ]; then
    echo "Error: Executable '$EXEC' not found. Please run 'make' first."
    exit 1
fi

echo "================================================================="
echo " Running Logic Propagator Test Suite with CUDA Profiling"
echo "================================================================="

PASSED=0
FAILED=0
TOTAL=0
GLOBAL_CSV="tests/sudoku/stats/${exec_name}_nsys_summary.csv"
echo "Test File,Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Kernel Name" > "$GLOBAL_CSV"

python3 compact_to_dimacs.py "$2"

for test_file in tests/sudoku/input/*.in; do
    [ -e "$test_file" ] || continue
    
    ((TOTAL++))
    filename=$(basename "$test_file")
    
    TMP_OUT="tests/sudoku/output/${filename}.tmp"
    FINAL_OUT="tests/sudoku/output/${filename}.out"
    RES_FILE="tests/sudoku/results/${filename}.res"
    NSYS_REP="tests/sudoku/output/${filename}_report"
    STATS_LOG="tests/sudoku/output/${filename}_nsys_stats.log"
    STATS_CSV="tests/sudoku/output/${filename}_nsys_stats.csv"
    
    nsys profile -t cuda --force-overwrite=true -o "$NSYS_REP" "$EXEC" < "$test_file" > "$TMP_OUT"
    nsys stats --force-report=true "${NSYS_REP}.nsys-rep" >> "$STATS_LOG"
    nsys stats --report=cuda_gpu_kern_sum --force-export=true --format=csv "${NSYS_REP}.nsys-rep" > "$STATS_CSV"
    
    if [ -s "$STATS_CSV" ]; then
        awk -v f="$filename" -F',' '$1 ~ /^[0-9]+\.?[0-9]*$/ {print f "," $0}' "$STATS_CSV" >> "$GLOBAL_CSV"
    fi
    
    python3 dimacs_to_compact.py "$TMP_OUT" "$FINAL_OUT"
    
    if diff -q -w "$FINAL_OUT" "$RES_FILE" > /dev/null; then
        echo -e "[\033[32mPASS\033[0m] $filename"
        ((PASSED++))
    else
        echo -e "[\033[31mFAIL\033[0m] $filename"
        echo "Output:"
        cat "$FINAL_OUT"
        echo "Expected:"
        cat "$RES_FILE"
        ((FAILED++))
    fi
done


rm -f tests/sudoku/output/*.tmp
rm -f tests/sudoku/output/*.out
rm -f tests/sudoku/input/*
rm -f tests/sudoku/results/*

rm -f tests/sudoku/output/*.nsys-rep
rm -f tests/sudoku/output/*.sqlite

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo "================================================================="
