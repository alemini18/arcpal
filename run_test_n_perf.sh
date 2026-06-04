#!/usr/bin/env bash

# Define paths
EXEC="$1"

# Ensure the executable exists
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

python3 compact_to_dimacs.py "$2"

# Loop through all .in files in the test directory
for test_file in tests/sudoku/input/*.in; do
    # Safety check in case the directory is empty
    [ -e "$test_file" ] || continue
    
    ((TOTAL++))
    filename=$(basename "$test_file")
    
    # Variabili per i file di output per maggiore chiarezza
    TMP_OUT="tests/sudoku/output/${filename}.tmp"
    FINAL_OUT="tests/sudoku/output/${filename}.out"
    RES_FILE="tests/sudoku/results/${filename}.res"
    NSYS_REP="tests/sudoku/output/${filename}_report"
    STATS_LOG="tests/sudoku/output/${filename}_nsys_stats.log"
    STATS_CSV="tests/sudoku/output/${filename}_nsys_stats.csv"
    #NCU_LOG="tests/sudoku/output/${filename}.ncu"
    
    # Esegue l'eseguibile tramite nsys
    # L'output standard del programma va in .tmp, mentre le statistiche di nsys (stderr) vanno in .log
    #
    nsys profile -t cuda --force-overwrite=true -o "$NSYS_REP" "$EXEC" < "$test_file" > "$TMP_OUT" 2> /dev/null
    nsys stats --force-report=true "${NSYS_REP}.nsys-rep" >> "$STATS_LOG" 2>&1
    nsys stats -r --force-report=true cuda_gpu_kern_sum --format csv "${NSYS_REP}.nsys-rep" > "$STATS_CSV"
    
    # 3. Accoda le statistiche al CSV globale, saltando l'intestazione e aggiungendo il nome del test
    if [ -s "$STATS_CSV" ]; then
        tail -n +2 "$STATS_CSV" | awk -v f="$filename" -F',' 'NF>0 {print f "," $0}' >> "$GLOBAL_CSV"
    fi
    #ncu --set full -o "$NCU_LOG" "$EXEC" < "$test_file" > "$TMP_OUT" 2> "$STATS_LOG"
    
    python3 dimacs_to_compact.py "$TMP_OUT" > "$FINAL_OUT"
    
    if diff -q -w "$FINAL_OUT" "$RES_FILE" > /dev/null; then
        echo -e "[\033[32mPASS\033[0m] $filename"
        echo -e "       \033[34m↳ Statistiche CUDA salvate in:\033[0m $STATS_LOG"
        ((PASSED++))
    else
        echo -e "[\033[31mFAIL\033[0m] $filename"
        echo "Output:"
        cat "$FINAL_OUT"
        echo "Expected:"
        cat "$RES_FILE"
        echo -e "       \033[34m↳ Statistiche CUDA salvate in:\033[0m $NCU_LOG"
        ((FAILED++))
    fi
done

# Clean up temp log
# ATTENZIONE: Modificato per NON eliminare i log .log appena generati e i file di input/risultati.
rm -f tests/sudoku/output/*.tmp
rm -f tests/sudoku/output/*.out
rm -f tests/sudoku/input/*
rm -f tests/sudoku/results/*
# Se non hai intenzione di aprire i report grafici nella GUI di Nsight Systems, 
# puoi decommentare le righe seguenti per risparmiare spazio su disco:
rm -f tests/sudoku/output/*.nsys-rep
rm -f tests/sudoku/output/*.sqlite

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo "================================================================="

# Exit with an error code if any test failed (useful for CI/CD pipelines)
if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi