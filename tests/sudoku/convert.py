import random
import sys

r = open("sudoku_9x9.in","r")
d = open(sys.argv[1],"r")

rules = r.read()

data = d.readlines(1700)


N = 9

idx = 0

for l in data:

    if "puzzle,solution" in l:
        continue

    out = open(f"tests/sudoku/instance1/sudoku_9x9_{idx}.in","w")
    out.write(rules)

    pos = 0
    #print(l)
    for c in l:
        if c == ',':
            break
        elif c != '0':
            out.write(f" {pos*N+int(c)}")
        pos+=1

    idx+=1
    out.write(" 0")
    out.close()

