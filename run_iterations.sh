#!/usr/bin/env bash

# Raccoglie le iterazioni del punto fisso di ogni variante, lette dalla prima riga dell'output,
# che ora ha la forma 's <esito> <iterazioni>'.
# Uso: ./run_iterations.sh [directory-delle-istanze]
# Senza argomenti usa tests/scaling/input, le stesse istanze di run_scaling.sh.
#
# Oltre a raccogliere il dato lo script lo verifica: la prima riga deve avere il campo nuovo e
# deve essere un numero, e il resto dell'output deve coincidere con quello della baseline seriale.

INPUT_DIR="${1:-tests/scaling/input}"
BUILD_DIR="build"
STATS_DIR="tests/scaling/stats"
REF="$BUILD_DIR/serial_naive"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found. Please run 'make' first."
    exit 1
fi

echo "================================================================="
echo " Counting Logic Propagator Fixpoint Iterations"
echo "================================================================="

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

mkdir -p "$STATS_DIR"

GLOBAL_CSV="$STATS_DIR/iterations.csv"
echo "Test File,Rules,Lits,Configuration,Status,Iterations" > "$GLOBAL_CSV"

TMP_OUT="$(mktemp)"
REF_OUT="$(mktemp)"

for test_file in "$INPUT_DIR"/*.in; do
    [ -e "$test_file" ] || continue

    filename=$(basename "$test_file")

    # Taglia dell'istanza letta dall'istanza stessa, non dal nome del file
    num_rules=$(awk 'NR==1 {print $3; exit}' "$test_file")
    num_lits=$(awk '$1 == "r" {l += $4} END {print l + 0}' "$test_file")

    echo "--- $filename ($num_rules regole, $num_lits letterali)"

    # Riferimento di correttezza: la baseline seriale, calcolata una volta sola per istanza
    HAS_REF=0
    if [ -x "$REF" ]; then
        "$REF" < "$test_file" > "$REF_OUT"
        HAS_REF=1
    fi

    for exec in "$BUILD_DIR"/*; do
        [ -f "$exec" ] && [ -x "$exec" ] || continue

        ((TOTAL++))
        exec_name=$(basename "$exec")

        "$exec" < "$test_file" > "$TMP_OUT"

        first=$(head -n 1 "$TMP_OUT")
        status=$(echo "$first" | awk '{print $2}')
        iterations=$(echo "$first" | awk '{print $3}')

        echo "$filename,$num_rules,$num_lits,$exec_name,$status,$iterations" >> "$GLOBAL_CSV"

        # Il campo nuovo deve esserci sempre ed essere un intero, anche sulle istanze rifiutate
        if ! echo "$first" | grep -qE '^s (SUCCESS|CONTRADICTION|ERROR) [0-9]+$'; then
            printf "[\033[31mFAIL\033[0m] %-22s prima riga senza il campo iterazioni: %s\n" "$exec_name" "$first"
            ((FAILED++))
        elif [ "$status" = "ERROR" ]; then
            printf "[\033[33mSKIP\033[0m] %-22s rifiutata dalla variante\n" "$exec_name"
            ((SKIPPED++))
        elif [ "$HAS_REF" -eq 0 ]; then
            printf "[\033[33mSKIP\033[0m] %-22s iterazioni: %-5s (nessun riferimento, manca '%s')\n" "$exec_name" "$iterations" "$REF"
            ((SKIPPED++))
        # Dalla prima riga si toglie il numero di iterazioni, che dipende dalla variante
        elif diff -q -w <(sed '1s/ [0-9]*$//' "$TMP_OUT") <(sed '1s/ [0-9]*$//' "$REF_OUT") > /dev/null; then
            printf "[\033[32mPASS\033[0m] %-22s iterazioni: %-5s (s %s)\n" "$exec_name" "$iterations" "$status"
            ((PASSED++))
        else
            printf "[\033[31mFAIL\033[0m] %-22s punto fisso diverso dalla baseline seriale\n" "$exec_name"
            echo "Output:   $first"
            echo "Expected: $(head -n 1 "$REF_OUT")"
            ((FAILED++))
        fi
    done
done

rm -f "$TMP_OUT" "$REF_OUT"

echo "================================================================="
echo " Summary: $PASSED / $TOTAL tests passed, $SKIPPED skipped."
echo " Written to $GLOBAL_CSV"
echo "================================================================="

if [ "$TOTAL" -eq 0 ]; then
    echo "Error: No instance found in '$INPUT_DIR', or no executable in '$BUILD_DIR'."
    exit 1
fi

[ "$FAILED" -eq 0 ]
