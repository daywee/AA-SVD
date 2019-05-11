import numpy as np
from math import hypot

np.set_printoptions(linewidth=200)
np.set_printoptions(precision=3)

def get_bidiagonal(size, random=True):
  np.random.seed(0)

  if random:
    M = np.random.rand(size, size)
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

def get_rot(a,b):
  """Compute matrix entries for Givens rotation."""
  """https://en.wikipedia.org/wiki/Givens_rotation"""
  r = hypot(a, b)
  c = a/r
  s = -b/r

  return (c,s)

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
npsvd = np.linalg.svd(A, full_matrices=True)
print('numpy reconstruction')
xxxxxx = npsvd[0][:,:5]
# print(xxxxxx)
# print(npsvd[0])
print(np.dot(npsvd[0] * npsvd[1], npsvd[2]))

# print(npsvd[0].dot(eyes.dot(npsvd[2].T)))

print('original')
print(A)

for iteration in range(1000):
  # print(A)
  T = get_T_matrix(A)
  # print('T',T)
  shift = get_closest_eigenvalue(T)
  yy = T[0,0] - shift
  zz = T[0,1]
  # print('shift', shift)
  # print('x', x)
  # print('rot', rot)

  U = None
  V = None

  size = 5
  for i in range(size-1):
    # x = np.array([A[i,i]**2-shift, A[i,i] * A[i,i+1]])
    rot = givens_rotation_matrix_entries(yy, zz)

    (c, s) = get_rot(yy, zz)
    rot = np.array([[c,s], [-s, c]])

    y = A[:, i:i+2]
    y = y.dot(rot)
    A[:, i:i+2] = y
    print(A)

    e = np.eye(size, dtype='float')
    e[i:i+2,i:i+2] = rot

    if U is None:
      U = e
    else:
      U = U.dot(e)

    # second part

    yy = A[i,i]
    zz = A[i+1, i]

    rot = givens_rotation_matrix_entries(yy, zz)

    (c, s) = get_rot(yy, zz)
    rot = np.array([[c,s], [-s, c]])

    # print(y)
    y = A[i:i+2, :]
    y = rot.T.dot(y)
    A[i:i+2, :] = y
    print(A)

    e = np.eye(size, dtype='float')
    e[i:i+2,i:i+2] = rot.T

    if V is None:
      V = e
    else:
      V = e.dot(V)

    if i < size-2:
      yy=A[i,i+1]
      zz=A[i,i+2]
    # print(y)
    # A[i:i+2, i:i+2] = y
    # print(A)


# T = A[3:4][3:4]
# T = A[np.ix_([3,4],[3,4])]

s = ''
for i in range(size):
  s += str(A[i,i])
  s += ','
print('my values:')
print(s)

print(A)

# print(np.linalg.eig(T))

print('results===============')
sings = []
for i in range(size):
  sings.append(A[i,i])
sings = np.array(sings)

print(U.dot(A).dot(V.T))
print(U.T.dot(A).dot(V))
print(V.dot(A).dot(U.T))
print(V.T.dot(A).dot(U))

# print(np.dot(U * sings, V.T))
# print(np.dot(U.T * sings, V))
# print(np.dot(V * sings, U.T))
# print(np.dot(V.T * sings, U))

