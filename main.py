import numpy as np
from math import hypot


def get_bidiagonal(size, random=True):
  np.random.seed(0)

  if random:
    M = np.random.rand(size, size, dtype='float')
  else:
    M = np.arange(1, size*size + 1, dtype='float').reshape((size, size))

  for i in range(0, size):
    for j in range(0, size):
      if (i != j and i + 1 != j):
        M[i][j] = 0

  return M

def get_closer_number(numbers, number):
  diffA = np.abs(numbers[0] - number)
  diffB = np.abs(numbers[1] - number)

  if diffA < diffB:
    return numbers[0]
  else:
    return numbers[1]

def apply_sign(a, b):
  s = np.sign(b)
  if s >= 0:
    if a >= 0:
      return a
    else:
      return -a
  else:
    if a < 0:
      return a
    else:
      return -a

def get_T_matrix(M):
  (rows, cols) = M.shape
  T2 = M[:,cols-2:cols]
  T1 = T2.T
  T = T1.dot(T2)

  return T

def get_closest_eigenvalue(T):
  lambdas = np.linalg.eig(T)[0]
  shift = get_closer_number(lambdas, T[0,0])

  return shift

def givens_rotation_matrix_entries(a, b):
  """Compute matrix entries for Givens rotation."""
  """https://en.wikipedia.org/wiki/Givens_rotation"""
  # r = hypot(a, b)
  # c = a/r
  # s = -b/r

  h = hypot(a,b)
  d = 1.0/h
  c = abs(a) * d
  s = apply_sign(d,a) * b
  r = apply_sign(1.0, a) * h

  return np.array([c, s, -s, c]).reshape((2,2))

A = get_bidiagonal(5, random=False)
print('numpy impl')
print(np.linalg.svd(A, full_matrices=True)[1])

for iteration in range(10000):
  # print(A)
  T = get_T_matrix(A)
  # print('T',T)
  shift = get_closest_eigenvalue(T)
  # print('shift', shift)
  x = np.array([A[0,0]**2-shift, A[0,0] * A[0,1]])
  # print('x', x)
  rot = givens_rotation_matrix_entries(x[0], x[1])
  # print('rot', rot)

  size = 5
  for i in range(size-1):
    c = A[i:i+2, i:i+2]
    c = rot.dot(c)
    # print(c)
    c = c.dot(rot.T)
    # print(c)
    cc = np.array(c)
    A[i:i+2, i:i+2] = c


# T = A[3:4][3:4]
# T = A[np.ix_([3,4],[3,4])]

print(A)

# print(np.linalg.eig(T))

