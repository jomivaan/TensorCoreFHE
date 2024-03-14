import numpy as np

A = np.zeros((256,256))
B = np.zeros((256,256))

for i in range(256):
    for j in range(256):
        A[i][j] = j % 2


for i in range(256):
    for j in range(256):
        B[i][j] = j % 2

C = np.matmul(A,B)
print(C)