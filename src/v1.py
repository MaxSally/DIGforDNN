import numpy as np
import z3

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])


# B = np.array([[z3.Real("i0"), z3.Real("i1"), z3.Real("i2")]])
B = np.array([[3, 5, 7]])

print(B.shape, A.shape)
print(B)
print(A)
v = B.dot(A)
print(v)
v = v + [7, 2]
print(v)
