#!/usr/bin/env bash

# Genera il dataset del benchmark di scala in tests/scaling/input.
# Le istanze sono di tre famiglie, riconoscibili dal prefisso del nome:
#   sudoku      i tre Sudoku 9x9, 16x16 e 25x25, con circa un terzo delle celle come indizi
#   synth_L008  regole corte, che entrano in un tile
#   synth_L200  regole lunghe, che un tile deve scorrere in piu' passate

INPUT_DIR="tests/scaling/input"

mkdir -p "$INPUT_DIR" tests/scaling/output tests/scaling/results tests/scaling/stats

# I riferimenti di correttezza valgono per le istanze precedenti, si rigenerano con loro
rm -f tests/scaling/results/*.res

python3 tests/sudoku.py 3 28 "$INPUT_DIR"
python3 tests/sudoku.py 4 90 "$INPUT_DIR"
python3 tests/sudoku.py 5 219 "$INPUT_DIR"

for rules in 500 1000 2000 4000 8000; do
    python3 tests/synthetic.py "$rules" 8 4 "$INPUT_DIR"
    python3 tests/synthetic.py "$rules" 200 4 "$INPUT_DIR"
done

echo "Generated $(ls "$INPUT_DIR"/*.in | wc -l) instances in $INPUT_DIR"
