import sys

d = open(sys.argv[1],"r")

if "s CONTRADICTION" in d.readline():
    print("This file contains a contradiction, not producing compact form")
    sys.exit(0)

atoms = d.readline().split()

atoms = atoms[1::]

N = 9

idx = 0
printed = False

#print(atoms)

for i in atoms:
    val = int(i)
    #print(val)
    if val > 0 and val != 730:
        print(val-(idx//9)*N,end="")
        printed = True
    if idx % 9 == 8:
        if not printed:
            print(0,end="")
        printed = False
    idx+=1

print("")


