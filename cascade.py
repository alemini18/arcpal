n = 10000

print("p " + str(2*n) + " " + str(n))

for i in range(0, n):
    print("r " + str(n+i+1) + " "  + str(n) + " " + str(n),end=" ")
    for j in range(1,n):
        print(str(j) + " 1", end=" ")
    print(str(n+i) + " 1")

print("a", end=" ")
for i in range(1,n+1):
    print(i,end=" ")
print("0")
