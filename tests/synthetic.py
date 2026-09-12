import sys

# Uso: python3 tests/synthetic.py [regole] [letterali] [strati] [directory]
# Genera un'istanza sintetica a strati in cui ogni regola ha esattamente L letterali di peso 1
# e bound L, quindi la testa e' vera se e solo se lo e' tutto il corpo.
# Le regole dello strato 0 leggono atomi gia' assegnati, quelle dello strato d leggono le teste
# dello strato d - 1: il punto fisso richiede percio' D + 1 iterazioni qualunque sia il numero
# di regole, e il numero di letterali per regola resta un parametro indipendente.
R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
L = int(sys.argv[2]) if len(sys.argv) > 2 else 200
D = int(sys.argv[3]) if len(sys.argv) > 3 else 4
OUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "tests/scaling/input"

W = max(1, R // D)
NUM_BASE = W + L - 1
regole = []

def base_atom(i):
    return i % NUM_BASE + 1

def head_atom(d, w):
    return NUM_BASE + d * W + w + 1

# Gli atomi di base pari compaiono positivi e quelli dispari negativi, sempre con lo stesso
# segno, cosi' l'assegnazione iniziale li soddisfa tutti ed esercita anche il caso lit < 0.
def base_lit(i):
    atom = base_atom(i)
    return atom if atom % 2 == 0 else -atom

for d in range(D):
    for w in range(W):
        regola = [str(head_atom(d, w)), str(L), str(L)]
        for j in range(L):
            lit = base_lit(w + j) if d == 0 else head_atom(d - 1, (w + j) % W)
            regola.append(str(lit))
            regola.append("1")
        regole.append(" ".join(regola))

with open(f"{OUT_DIR}/synth_L{L:03d}_r{W * D:06d}.in", "w") as f:
    f.write("p " + str(NUM_BASE + D * W) + " " + str(len(regole)) + "\n")
    for r in regole:
        f.write("r " + r + "\n")
    f.write("a")
    for i in range(NUM_BASE):
        f.write(" " + str(base_lit(i)))
    f.write(" 0\n")
