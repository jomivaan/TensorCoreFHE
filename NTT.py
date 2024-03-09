
from sympy.ntheory.residue_ntheory import nthroot_mod
import numpy as np

def bit_reverse(lst):
    n = len(lst)
    max_bits = len(bin(n - 1)) - 2  # Number of bits required to represent indices
    return [lst[int(format(i, '0' + str(max_bits) + 'b')[::-1], 2)] for i in range(n)]

roots = nthroot_mod(1, 16, 1152921504606846577, all_roots=True)
# print(roots)
# print(bit_reverse(roots))
# OpenFHEroots_2 = [1, 85100811348556653]
# OpenFHEroots_4 = [1, 85100811348556653, 8713384291351227, 164055430638770543]
# OpenFHEroots_8 = [1, 203129564803191956, 124174945512978066, 279516991860397382, 14110678920874275, 192014686236329522, 28133603095387109, 179246385905362438 ]
# OpenFHEroots_16 = [1, 203129564803191956, 124174945512978066, 279516991860397382, 14110678920874275, 192014686236329522, 28133603095387109, 179246385905362438, 3680809762999808, 74826503740590966, 84598026171650641, 260901429737366068, 47832758282793489, 235552869988995991, 119464826350785041, 264185975921425219]
# OpenFHEroots_32 = [1, 203129564803191956, 124174945512978066, 279516991860397382, 274119697230874334, 96215689915419087, 260096773056361500, 108983990246386171, 213403872411157643, 3680809762999808, 27328946414382541, 84598026171650641, 235552869988995991, 240397617868955120, 264185975921425219, 168765549800963568, 32991918180745791, 226681119651494510, 236611345769281145, 83692195984946206, 249683523330574751, 114511695462353866, 195877461242637253, 58958849189422535, 181088768555838315, 111181120054800642, 56617894171396881, 92618928701795850, 70498368981727956, 140776271856403595, 186903532642365870, 58978252375354895 ]

# OpenFHErootsNO_2 = bit_reverse(OpenFHEroots_2)
# OpenFHErootsNO_4 = bit_reverse(OpenFHEroots_4)
# OpenFHErootsNO_8 = bit_reverse(OpenFHEroots_8)
# OpenFHErootsNO_16 = bit_reverse(OpenFHEroots_16)
# OpenFHErootsNO_32 = bit_reverse(OpenFHEroots_32)
# input = np.matrix([[248624671961015766, 104064564287205047, 186556200519391708, 183365194556609938], [203434914383328712, 56598977939943515, 26767075655198788, 274583609827135228] ])
modulus = 288230376151748609
input_NO = [248624671961015766, 104064564287205047, 186556200519391708, 183365194556609938, 203434914383328712, 56598977939943515, 26767075655198788, 274583609827135228 ]
input = np.matrix([[248624671961015766, 104064564287205047],[186556200519391708, 183365194556609938],[203434914383328712, 56598977939943515],[26767075655198788, 274583609827135228]])
print(input)
# print(OpenFHErootsNO_2)
# print(25*"*")
# print(OpenFHErootsNO_4)
# print(25*"*")
# print(OpenFHErootsNO_8)
# print(25*"*")
# print(OpenFHErootsNO_16)
# print(30*"*")
# print(OpenFHErootsNO_32)
# print(50*"*")

# input = np.matrix([[1, 2],[ 3, 4], [5, 6],[ 7, 8] ])
input = np.matrix([[0, 1],[ 2, 3], [4, 5],[ 6, 7] ])
modulus = 5409
input_NO = [0,1, 2, 3, 4, 5, 6, 7 ]

root8 = 15
my_roots_8 = []
print("***********"+"roots"+"***********")
for i in range(2*8):
    if i == 0:
        my_roots_8.append(1)
    else:
        my_roots_8.append((my_roots_8[i-1]*root8 ))

my_roots_2 = []
for i in range(2*2):
    if i == 0:
        my_roots_2.append(1)
    else:
        my_roots_2.append((my_roots_2[i-1]*my_roots_8[4]) )

my_roots_4 = []
for i in range(2*4):
    if i == 0:
        my_roots_4.append(1)
    else:
        my_roots_4.append(my_roots_4[i-1]*my_roots_8[2] )

for i in range(0, len(my_roots_8)):
    my_roots_8[i] = my_roots_8[i] % modulus

for i in range(0, len(my_roots_4)):
    my_roots_4[i] = my_roots_4[i] % modulus

for i in range(0, len(my_roots_2)):
    my_roots_2[i] = my_roots_2[i] % modulus

print(my_roots_8)
print(my_roots_4)
print(my_roots_2)
print("***********"+"matrices"+"***********")
N1 = 4
N2 = 2
W1 = np.zeros((N1,N1))
W2 = np.zeros((N1,N2))
W3 = np.zeros((N2,N2))
for i in range(0,N1):
    for j in range(0,N1):
        W1[i][j] = my_roots_4[(2*i*j + j) % (2*N1)]
 
print(W1)

print(50*"*")
for i in range(0,N1):
    for j in range(0,N2):
        W2[i][j] = my_roots_8[(2*i*j + j) % (2*N1*N2)]
        

print(W2)
print(50*"*")
for i in range(0,N2):
    for j in range(0,N2):
        W3[i][j] = my_roots_2[(2*i*j) % (2*N2)]


print(W3)
print(50*"*"+ "matrix input"+ 50*"*")
print(input)

print(50*"*"+ "start of tensor ntt (incorrect!!)"+ 50*"*")


inputX1 = np.matmul(W1,input)
inputX1 = inputX1 % modulus

W2mult = np.multiply(inputX1, W2)
W2mult = W2mult % modulus
W3mult = np.matmul(W2mult, W3)
W3mult = W3mult % modulus
print(75*"*")
print(W3mult)
print(75*"*")
print("Now radix 2 NTT (correct!! and exactly the same as OpenFHE)")

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
print(input_NO)