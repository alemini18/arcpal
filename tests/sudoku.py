K = 3
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

with open(f"sudoku_{K * K} x {K * K}.in", "w") as f:
    f.write("p " + str(N * N * N + 2) + " " + str(len(regole)) + "\n")
    for r in regole:
        f.write("r " + r + "\n")
    f.write("a " + str(N * N * N + 1) + " " + str(-(N * N * N + 2)) + "\n")