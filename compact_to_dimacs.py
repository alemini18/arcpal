import random
import sys

r = open("sudoku_9x9.in","r")
d = open(sys.argv[1],"r")

rules = r.read()

data = d.readlines()


N = 9

idx = 0

for l in data:

    if "puzzle,solution" in l:
        continue

    out = open(f"tests/sudoku/input/sudoku_9x9_{idx}.in","w")
    res = open(f"tests/sudoku/results/sudoku_9x9_{idx}.in.res","w")
    out.write(rules)

    pos = 0
    #print(l)
    flag = False
    for c in l:
        if c == ',':
            flag = True
        elif flag == False and c != '0':
            out.write(f" {pos*N+int(c)}")
        elif flag == True:
            res.write(str(c))
        pos+=1
            

    idx+=1
    out.write(" 0")
    out.close()

