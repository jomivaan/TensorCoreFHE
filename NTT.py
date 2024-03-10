
from sympy.ntheory.residue_ntheory import nthroot_mod
import numpy as np
np.seterr(all='raise') 

def bit_reverse(lst):
    n = len(lst)
    max_bits = len(bin(n - 1)) - 2  # Number of bits required to represent indices
    return [lst[int(format(i, '0' + str(max_bits) + 'b')[::-1], 2)] for i in range(n)]

roots = nthroot_mod(1, 16, 1152921504606846577, all_roots=True)

# input = np.matrix([[248624671961015766, 104064564287205047, 186556200519391708, 183365194556609938], [203434914383328712, 56598977939943515, 26767075655198788, 274583609827135228] ])
modulus = 288230376151748609
input_NO = [248624671961015766, 104064564287205047, 186556200519391708, 183365194556609938, 203434914383328712, 56598977939943515, 26767075655198788, 274583609827135228 ]
# input = np.matrix([[248624671961015766, 104064564287205047],[186556200519391708, 183365194556609938],[203434914383328712, 56598977939943515],[26767075655198788, 274583609827135228]])
root8 = 14110678920874275


# modulus = 288
# input_NO = [0,1, 2, 3, 4, 5, 6, 7 ]
# root8 = 5

my_roots_8 = []
print("***********"+"roots"+"***********")
for i in range(2*8):
    if i == 0:
        my_roots_8.append(1)
    else:
        my_roots_8.append((my_roots_8[i-1]*root8 )% modulus)

my_roots_2 = []
for i in range(2*2):
    if i == 0:
        my_roots_2.append(1)
    else:
        my_roots_2.append((my_roots_2[i-1]*my_roots_8[4]) % modulus )

my_roots_4 = []
for i in range(2*4):
    if i == 0:
        my_roots_4.append(1)
    else:
        my_roots_4.append((my_roots_4[i-1]*my_roots_8[2]) % modulus )

# for i in range(0, len(my_roots_8)):
#     my_roots_8[i] = my_roots_8[i] % modulus

# for i in range(0, len(my_roots_4)):
#     my_roots_4[i] = my_roots_4[i] % modulus

# for i in range(0, len(my_roots_2)):
#     my_roots_2[i] = my_roots_2[i] % modulus

print(my_roots_8)
print(my_roots_4)
print(my_roots_2)
print("***********"+"matrices"+"***********")
N1 = 4
N2 = 2
input = np.zeros((N1,N2),dtype=np.int64)
W1 = np.zeros((N1,N1),dtype=np.int64)
W2 = np.zeros((N1,N2),dtype=np.int64)
W3 = np.zeros((N2,N2),dtype=np.int64)
# input = np.zeros((N1,N2))
# W1 = np.zeros((N1,N1))
# W2 = np.zeros((N1,N2))
# W3 = np.zeros((N2,N2))
for i in range(0,N1):
    for j in range(0,N2):
        input[i][j] = input_NO[(N2*i + j) ]


for i in range(0,N1):
    for j in range(0,N1):
        W1[i][j] = my_roots_4[(2*i*j + j) % (2*N1)]

print(50*"*"+ "W1"+ 50*"*")
print(W1)

print(50*"*")
for i in range(0,N1):
    for j in range(0,N2):
        W2[i][j] = my_roots_8[(2*i*j + j) % (2*N1*N2)]
        
print(50*"*"+ "W2"+ 50*"*")
print(W2)
print(50*"*")
for i in range(0,N2):
    for j in range(0,N2):
        W3[i][j] = my_roots_2[(2*i*j) % (2*N2)]

print(50*"*"+ "W3"+ 50*"*")
print(W3)
print(50*"*"+ "matrix input"+ 50*"*")
print(input)

print(50*"*"+ "start of tensor ntt (incorrect!!)"+ 50*"*")


inputX1 = np.matmul(W1,input, dtype=np.int64)
inputX1 = inputX1 % modulus

print(25*"*"+ "W1 x input"+25*"*")
print(inputX1)
print(75*"*")
W2mult = np.multiply(inputX1, W2,dtype=np.int64)
W2mult = W2mult % modulus
print(25*"*"+ "W2mult"+25*"*")
print(W2mult)
print(75*"*")
W3mult = np.matmul(W2mult, W3, dtype=np.int64) 
W3mult = W3mult % modulus
print(25*"*"+ "W3mult"+25*"*")
print((W3mult.flatten()).tolist())
print(75*"*")

# result0 = input_NO[0] % modulus 
# print(result0)
# result1 = (input_NO[1]*my_roots_8[15]) % modulus
# print(result1)
# result2 = (input_NO[2]*my_roots_8[14]) % modulus
# print(result2)
# result3 = (input_NO[3]*my_roots_8[13]) % modulus
# print(result3)
# result4 = (input_NO[4]*my_roots_8[12]) % modulus
# print(result4)
# result5 = (input_NO[5]*my_roots_8[11]) % modulus
# print(result5)
# result6 = (input_NO[6]*my_roots_8[10]) % modulus
# print(result6)
# result7 = (input_NO[7]*my_roots_8[9]) % modulus
# print(result7)
# print("A7 By using only my_roots_8 is the sum of the above")
# result = input_NO[0] % modulus 
# result += input_NO[1]*my_roots_8[1] % modulus
# result += input_NO[2]*my_roots_8[2] % modulus
# result += input_NO[3]*my_roots_8[3] % modulus
# result += input_NO[4]*my_roots_8[4] % modulus
# result += input_NO[5]*my_roots_8[5] % modulus
# result += input_NO[6]*my_roots_8[6] % modulus
# result += +input_NO[7]*my_roots_8[7] % modulus
# result = result % modulus
# print("A0 By using only my_roots_8: "+str(result))
# result = (input_NO[0] + input_NO[2]*my_roots_4[7]+input_NO[4]*my_roots_4[6]+input_NO[6]*my_roots_4[5] +  my_roots_2[2]*my_roots_8[7]*(input_NO[1]+ input_NO[3]*my_roots_4[7]+ input_NO[5]*my_roots_4[6]+ input_NO[7]*my_roots_4[5])  ) % modulus
# print("A7 By using all: "+str(result))

# result1 = (input[0][0]*W1[3][0] + input[1][0]*W1[3][1]+input[2][0]*W1[3][2]+input[3][0]*W1[3][3] )% modulus
# result2 =  (input[0][1]*W1[3][0] + input[1][1]*W1[3][1]+input[2][1]*W1[3][2]+input[3][1]*W1[3][3] )% modulus

# result3 = result1*W2[3][0]
# result4 = result2*W2[3][1]

# result5 = result3*W3[0][1]+result4*W3[1][1]
# result5 = result5 % modulus
# print("A7 By using matrices "+str(result5))
print("Now radix 2 NTT (correct!! and exactly the same as OpenFHE)")


# result = (input_NO[0]+) % modulus
# print("By using  my_roots_8+4: "+str(result))

m = 1  # Initialize m
n = 8
t = n
RoU_array_BO = bit_reverse(my_roots_8[0:8])
output = [0,0,0,0,0,0,0,0]
while m < n:
    t = t >> 1  # Integer division by 2
    for i in range(m):
        j1 = 2 * i * t
        j2 = j1 + t - 1
        S = RoU_array_BO[m+i]
        for j in range(j1, j2 + 1):
            U = input_NO[j ]
            V = input_NO[j+t] * S
            input_NO[j] = (U + V) % modulus
            input_NO[j + t] = (U - V) % modulus
       

    m = 2 * m  # Double the value of m
print(50*"*")
print(bit_reverse(input_NO))