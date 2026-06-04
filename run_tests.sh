#!/usr/bin/env bash

# Define paths
EXEC="$1"
exec_name=$(basename "$EXEC")

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

# 1. Definisci i percorsi dei CSV globali e rimuovi i vecchi di run precedenti
GLOBAL_KERN_CSV="tests/sudoku/output/${exec_name}_nsys_kernels_summary.csv"
GLOBAL_MEM_CSV="tests/sudoku/output/${exec_name}_nsys_memory_summary.csv"
rm -f "$GLOBAL_KERN_CSV" "$GLOBAL_MEM_CSV"

# Loop through all .in files in the test directory
for test_file in tests/sudoku/input/*.in; do
    # Safety check in case the directory is empty
    [ -e "$test_file" ] || continue
    
    ((TOTAL++))
    filename=$(basename "$test_file")
    
    # Variabili per i file di output
    TMP_OUT="tests/sudoku/output/${filename}.tmp"
    FINAL_OUT="tests/sudoku/output/${filename}.out"
    RES_FILE="tests/sudoku/results/${filename}.res"
    NSYS_REP="tests/sudoku/output/${filename}_report"
    STATS_LOG="tests/sudoku/output/${filename}_nsys_stats.log"
    
    STATS_KERN_CSV="tests/sudoku/output/${filename}_nsys_kernels.csv"
    STATS_MEM_CSV="tests/sudoku/output/${filename}_nsys_memory.csv"
    
    # Rimuove vecchi export SQLite
    rm -f "${NSYS_REP}.sqlite"
    
    # Esegue l'eseguibile tramite nsys
    nsys profile -t cuda --force-overwrite=true -o "$NSYS_REP" "$EXEC" < "$test_file" > "$TMP_OUT" 2> /dev/null
    
    # Salva il log testuale classico (comprensivo di tutto)
    nsys stats --force-export=true "${NSYS_REP}.nsys-rep" >> "$STATS_LOG" 2>&1
    
    # 2. Estrae separatamente i due report in formato CSV
    nsys stats --force-export=true --report=cuda_gpu_kern_sum --format=csv "${NSYS_REP}.nsys-rep" > "$STATS_KERN_CSV" 2> /dev/null
    nsys stats --force-export=true --report=cuda_gpu_mem_time_sum --format=csv "${NSYS_REP}.nsys-rep" > "$STATS_MEM_CSV" 2> /dev/null
    
    # 3. Processa e accoda i dati dei KERNEL
    if [ -s "$STATS_KERN_CSV" ]; then
        # Se il CSV globale non esiste, crea dinamicamente l'header prendendolo da nsys
        if [ ! -f "$GLOBAL_KERN_CSV" ]; then
            awk -F',' '$1 ~ /^Time/ {print "Test File," $0}' "$STATS_KERN_CSV" > "$GLOBAL_KERN_CSV"
        fi
        # Accoda solo le righe con i dati numerici
        awk -v f="$filename" -F',' '$1 ~ /^[0-9]+\.?[0-9]*$/ {print f "," $0}' "$STATS_KERN_CSV" >> "$GLOBAL_KERN_CSV"
    fi
    
    # 4. Processa e accoda i dati della MEMORIA (Trasferimenti)
    if [ -s "$STATS_MEM_CSV" ]; then
        # Se il CSV globale della memoria non esiste, crea l'header dinamicamente
        if [ ! -f "$GLOBAL_MEM_CSV" ]; then
            awk -F',' '$1 ~ /^Time/ {print "Test File," $0}' "$STATS_MEM_CSV" > "$GLOBAL_MEM_CSV"
        fi
        # Accoda solo le righe con i dati numerici
        awk -v f="$filename" -F',' '$1 ~ /^[0-9]+\.?[0-9]*$/ {print f "," $0}' "$STATS_MEM_CSV" >> "$GLOBAL_MEM_CSV"
    fi
    
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
        echo -e "       \033[34m↳ Statistiche CUDA salvate in:\033[0m $STATS_LOG"
        ((FAILED++))
    fi
done

# Pulizia dei file temporanei e parziali
rm -f tests/sudoku/output/*.tmp
rm -f tests/sudoku/output/*.out
rm -f tests/sudoku/input/*
rm -f tests/sudoku/results/*
rm -f tests/sudoku/output/*.nsys-rep
rm -f tests/sudoku/output/*.sqlite
rm -f tests/sudoku/output/*_nsys_kernels.csv
rm -f tests/sudoku/output/*_nsys_memory.csv

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed."
echo " -> Kernels CSV saved in: $GLOBAL_KERN_CSV"
echo " -> Memory CSV saved in:  $GLOBAL_MEM_CSV"
echo "================================================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
else
    exit 0
fi