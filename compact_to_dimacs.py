import sys

rules_file = open("sudoku_9x9.in","r")
data_file = open(sys.argv[1],"r")

rules = rules_file.read()
rules_file.close()

data = data_file.readlines()
data_file.close()

N = 9
idx = 0

for line in data:

    if "puzzle,solution" in line:
        continue

    out = open(f"tests/sudoku/input/sudoku_9x9_{idx}.in","w")
    res = open(f"tests/sudoku/results/sudoku_9x9_{idx}.in.res","w")
    out.write(rules)

    pos = 0
    end = False
    for char in line:
        if char == ',':
            end = True
        elif end == False and char != '0':
            out.write(f" {pos * N + int(char)}")
        elif end == True:
            res.write(str(char))
        pos += 1
            
    idx += 1
    out.write(" 0")
    out.close()

