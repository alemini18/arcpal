import sys

data = open(sys.argv[1],"r")
out = open(sys.argv[2], "w")

if "s CONTRADICTION" in data.readline():
    print("This file contains a contradiction, not producing compact form")
    sys.exit(0)

atoms = data.readline().split()
data.close()

atoms = atoms[1::]

N = 9

idx = 0
printed = False

for i in atoms:
    val = int(i)
    if val > 0 and val != 730:
        out.write(str(val - (idx // 9) * N))
        printed = True
    if idx % 9 == 8:
        if not printed:
            out.write("0")
        printed = False
    idx += 1

out.write("\n")
out.close()


