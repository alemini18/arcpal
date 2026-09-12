#!/usr/bin/env bash

EXEC="$1"

exec_name="$(basename "$EXEC")"

INPUT_DIR="tests/scaling/input"
OUTPUT_DIR="tests/scaling/output"
RESULT_DIR="tests/scaling/results"

if [ ! -f "$EXEC" ]; then
    echo "Error: Executable '$EXEC' not found. Please run 'make' first."
    exit 1
fi

echo "================================================================="
echo " Running Logic Propagator Scaling Suite with CUDA Profiling"
echo "================================================================="

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

# Meno ripetizioni che sui Sudoku: qui le istanze sono molte volte piu' grandi
RUNS=3

HAS_NSYS=1
if ! command -v nsys > /dev/null 2>&1; then
    echo "Warning: 'nsys' not found, collecting wall-clock time only."
    HAS_NSYS=0
fi

GLOBAL_CSV="tests/scaling/stats/${exec_name}_scaling.csv"
echo "Test File,Rules,Lits,Run,Report,Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name" > "$GLOBAL_CSV"

for test_file in "$INPUT_DIR"/*.in; do
    [ -e "$test_file" ] || continue

    ((TOTAL++))
    filename=$(basename "$test_file")

    # Taglia dell'istanza letta dall'istanza stessa, non dal nome del file
    num_rules=$(awk 'NR==1 {print $3; exit}' "$test_file")
    num_lits=$(awk '$1 == "r" {l += $4} END {print l + 0}' "$test_file")

    TMP_OUT="$OUTPUT_DIR/${filename}.out"
    RES_FILE="$RESULT_DIR/${filename}.res"
    NSYS_REP="$OUTPUT_DIR/${filename}_report"
    STATS_CSV="$OUTPUT_DIR/${filename}_nsys_stats.csv"

    # Riferimento di correttezza: la baseline seriale, calcolata una volta sola per istanza
    if [ ! -f "$RES_FILE" ] && [ -x build/serial_naive ]; then
        build/serial_naive < "$test_file" > "$RES_FILE"
    fi

    # Giro di riscaldamento, che crea il contesto CUDA e rivela le istanze rifiutate per taglia
    "$EXEC" < "$test_file" > "$TMP_OUT"

    if head -n 1 "$TMP_OUT" | grep -q "s ERROR"; then
        echo -e "[\033[33mSKIP\033[0m] $filename ($num_rules regole, rifiutata dalla variante)"
        ((SKIPPED++))
        continue
    fi

    for run in $(seq 1 "$RUNS"); do

        # Tempo di parete senza profiler, confrontabile con la baseline seriale
        start_ns=$(date +%s%N)
        "$EXEC" < "$test_file" > "$TMP_OUT"
        end_ns=$(date +%s%N)

        echo "$filename,$num_rules,$num_lits,$run,wall,100.0,$((end_ns - start_ns)),1,,,,,,total" >> "$GLOBAL_CSV"

        [ "$HAS_NSYS" -eq 1 ] || continue

        nsys profile -t nvtx,cuda --force-overwrite=true -o "$NSYS_REP" "$EXEC" < "$test_file" > /dev/null

        for report in nvtx_pushpop_sum nvtx_sum cuda_gpu_kern_sum; do
            nsys stats --report="$report" --force-export=true --format=csv "${NSYS_REP}.nsys-rep" > "$STATS_CSV"

            if [ -s "$STATS_CSV" ]; then
                awk -v f="$filename" -v nr="$num_rules" -v nl="$num_lits" -v r="$run" -v rep="$report" -F',' '$1 ~ /^[0-9]+\.?[0-9]*$/ {print f "," nr "," nl "," r "," rep "," $0}' "$STATS_CSV" >> "$GLOBAL_CSV"
            fi
        done
    done

    if [ ! -f "$RES_FILE" ]; then
        echo -e "[\033[33mSKIP\033[0m] $filename (nessun riferimento, manca 'build/serial_naive')"
        ((SKIPPED++))
    # Dalla prima riga si toglie il numero di iterazioni, che dipende dalla variante
    elif diff -q -w <(sed '1s/ [0-9]*$//' "$TMP_OUT") <(sed '1s/ [0-9]*$//' "$RES_FILE") > /dev/null; then
        echo -e "[\033[32mPASS\033[0m] $filename ($num_rules regole, $num_lits letterali)"
        ((PASSED++))
    else
        # L'output di queste istanze e' lungo migliaia di atomi, se ne stampa solo l'esito
        echo -e "[\033[31mFAIL\033[0m] $filename"
        echo "Output:   $(head -n 1 "$TMP_OUT")"
        echo "Expected: $(head -n 1 "$RES_FILE")"
        ((FAILED++))
    fi
done

rm -f "$OUTPUT_DIR"/*.out
rm -f "$OUTPUT_DIR"/*.csv
rm -f "$OUTPUT_DIR"/*.nsys-rep
rm -f "$OUTPUT_DIR"/*.sqlite

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed, $SKIPPED skipped."
echo "================================================================="

if [ "$TOTAL" -eq 0 ]; then
    echo "Error: No instance found in '$INPUT_DIR'. Please run 'tests/generate_scaling.sh' first."
    exit 1
fi

[ "$FAILED" -eq 0 ]
