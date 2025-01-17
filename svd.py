import numpy as np
from math import hypot, sqrt

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

def get_random_matrix(size):
  return np.random.rand(size, size)

def find_q(B, epsilon):
  (m, n) = B.shape
  q = 0

  for i in range(n):
    x = n - 1 - i
    y = n - 1 - i - 1

    if x == 0:
      return q + 1
    if np.abs(B[y, x]) < epsilon:
      q += 1
    else:
      return q

def find_p(B, q, epsilon):
  (m, n) = B.shape
  p = n - q

  for i in range(q, n):
    x = n - 1 - i
    y = n - 1 - i - 1

    if x == 0:
      return p - 1
    if np.abs(B[y, x]) >= epsilon:
      p -= 1
    else:
      return p - 1

def get_closer_number(numbers, number):
  diff_a = np.abs(numbers[0] - number)
  diff_b = np.abs(numbers[1] - number)

  if diff_a < diff_b:
    return numbers[0]
  else:
    return numbers[1]

def get_givens_rotation_matrix(a, b):
  """Compute matrix entries for Givens rotation."""
  """https://en.wikipedia.org/wiki/Givens_rotation"""
  if b == 0.0:
    c = 1.0
    s = 0.0
    r = a
  else:
    r = hypot(a, b)
    c = a / r
    s = -b / r

  return np.array([[c, s], [-s, c]])

def svd_internal(B):
  (m, n) = B.shape

  T = B.T.dot(B)
  T_trailing = T[m - 2:m,n - 2:n]

  eigenvalues = np.linalg.eig(T_trailing)[0]
  shift = get_closer_number(eigenvalues, T_trailing[1, 1])

  y = T[0, 0] - shift
  z = T[0, 1]

  U = np.identity(n)
  V = np.identity(m)

  for k in range(n - 1):
    rotation_matrix = get_givens_rotation_matrix(y, z)
    x = B[:, k:k + 2]
    x = x.dot(rotation_matrix)
    B[:, k:k + 2] = x

    V_rotation_matrix = np.identity(n)
    V_rotation_matrix[k:k + 2, k:k + 2] = rotation_matrix
    V = V.dot(V_rotation_matrix)

    y = B[k, k]
    z = B[k + 1, k]

    rotation_matrix = get_givens_rotation_matrix(y, z)
    rotation_matrix = rotation_matrix.T
    x = B[k:k + 2, :]
    x = rotation_matrix.dot(x)
    B[k:k + 2, :] = x

    U_rotation_matrix = np.identity(m)
    U_rotation_matrix[k:k + 2, k:k + 2] = rotation_matrix
    U = U_rotation_matrix.dot(U)

    if (k < n - 1):
      y = B[k, k + 1]
      z = 0.0 if k + 2 == n else B[k, k + 2]

  return (U, B, V)


def svd(M, epsilon, max_iterations=10000):
  B = M.copy()
  (m, n) = B.shape

  if m != n:
    raise Exception('Dimension must be the same')

  U = np.identity(m)
  V = np.identity(n)

  q = 0
  iteration = 0
  while q < n and iteration < max_iterations:
    iteration += 1
    q = find_q(B, epsilon)
    p = find_p(B, q, epsilon)

    if q < n:
      b22_size = n - q - p
      b22 = B[p:p + b22_size + 1, p:p + b22_size + 1]

      (u, S, v) = svd_internal(b22)

      B[p:p + b22_size + 1, p:p + b22_size + 1] = S

      pu = np.identity(b22_size + p + q)
      pu[p:p + b22_size + 1, p:p + b22_size + 1] = u

      pv = np.identity(b22_size + p + q)
      pv[p:p + b22_size + 1, p:p + b22_size + 1] = v

      U = pu.dot(U)
      V = V.dot(pv)

  return (U.T, B, V)

# A = np.array([[4, 3, 0, 2], [2, 1, 2, 1], [4, 4, 0, 3], [5, 6, 1, 3]], dtype='float64')
# A = get_bidiagonal(100, random=True)
A = get_random_matrix(100)
print('original')
print(A)
print()

s = np.linalg.svd(A, full_matrices=True)
print('numpy svd:')
print(s[1])
print()
(U, S, V) = svd(A, np.finfo(np.float).eps, 100)

print('computed singular values:')
s = ''
for i in range(S.shape[0]):
  s += str(S[i,i]) + ', '
s = s[:len(s)-2] # strip last comma
print(s)
print()

print('reconstruction:')
R = U.dot(S).dot(V.T)
print(R, '\n')

print('Reconstruction equals to original')
print(np.allclose(A, R))
