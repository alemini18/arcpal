import sys

data = open(sys.argv[1],"r")
out = open(sys.argv[2], "w")

if "s CONTRADICTION" in data.readline():
    print("This file contains a contradiction, not producing compact form")
    sys.exit(0)

atoms = data.readline().split()
data.close()

atoms = atoms[1::]

K = 3
N = K * K

idx = 0
printed = False

for i in atoms:
    val = int(i)
    if val > 0 and val < N * N * N + 1:
        out.write(str(val - (idx // N) * N))
        printed = True
    if idx % N == N - 1:
        if not printed:
            out.write("0")
        printed = False
    idx += 1

out.write("\n")
out.close()


