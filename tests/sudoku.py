import sys
import random

# Uso: python3 tests/sudoku.py [K] [indizi] [directory]
# K = 3 genera il Sudoku 9x9, K = 4 il 16x16, K = 5 il 25x25.
# Con indizi = 0 la riga 'a' resta aperta e viene completata da compact_to_dimacs.py con il
# puzzle preso dal dataset; con indizi > 0 lo script chiude l'istanza scrivendo lui stesso gli
# indizi, presi da una soluzione valida costruita analiticamente.
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CLUES = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "."
N = K * K
regole = []

H_TRUE = N * N * N + 1
H_FALSE = N * N * N + 2

def get_var(r, c, v):
    return (r - 1) * (N * N) + (c - 1) * N + (v - 1) + 1

def at_least_one(vars):
    k = len(vars)
    regola = [str(H_TRUE), "1", str(k)]
    for var in vars:
        regola.append(str(var))
        regola.append("1")
    regole.append(" ".join(regola))

def at_most_one(vars):
    k = len(vars)
    regola = [str(H_FALSE), "2", str(k)]
    for var in vars:
        regola.append(str(var))
        regola.append("1")
    regole.append(" ".join(regola))

def exactly_one(vars):
    at_least_one(vars)
    at_most_one(vars)

for r in range(1, N + 1):
    for c in range(1, N + 1):
        vars = [get_var(r, c, v) for v in range(1, N + 1)]
        exactly_one(vars)

for r in range(1, N + 1):
    for v in range(1, N + 1):
        vars = [get_var(r, c, v) for c in range(1, N + 1)]
        exactly_one(vars)

for c in range(1, N + 1):
    for v in range(1, N + 1):
        vars = [get_var(r, c, v) for r in range(1, N + 1)]
        exactly_one(vars)

for br in range(K):
    for bc in range(K):
        for v in range(1, N + 1):
            vars = []
            for i in range(1, K + 1):
                for j in range(1, K + 1):
                    r = br * K + i
                    c = bc * K + j
                    vars.append(get_var(r, c, v))
            exactly_one(vars)

def solution(r, c):
    return (K * ((r - 1) % K) + (r - 1) // K + (c - 1)) % N + 1

with open(f"{OUT_DIR}/sudoku_{N}x{N}.in", "w") as f:
    f.write("p " + str(N * N * N + 2) + " " + str(len(regole)) + "\n")
    for r in regole:
        f.write("r " + r + "\n")
    f.write("a " + str(N * N * N + 1) + " " + str(-(N * N * N + 2)) + "\n")
    if CLUES > 0:
        for cella in sorted(random.Random(0).sample(range(N * N), CLUES)):
            r = cella // N + 1
            c = cella % N + 1
            f.write(" " + str(get_var(r, c, solution(r, c))))
        f.write(" 0\n")