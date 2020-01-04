import numpy as np

# resources
# https://github.com/emchinn/Bidiagonalization/blob/master/Golub-Kahan.ipynb
# http://www.math.iit.edu/~fass/477577_Chapter_12.pdf

def sign(x):
  return 1 if x >= 0 else -1

def get_e(length, i):
  e = np.zeros(length)
  e[i] = 1.0
  return e

def gk(A):
  (m, n) = A.shape

  if m < n:
    raise Exception('m cannot be lesser than n')

  # C = np.zeros((n,m))
  # S = np.ones(m)
  # R = np.ones(n)

  # for k in range(n):
  #   x = A[k:, k]

  #   e1 = np.zeros(len(x))
  #   e1[0] = 1.0

  #   u = x + sign(x[0]) * np.linalg.norm(x) * e1
  #   u = u / np.linalg.norm(u)

  #   c = (u.T * A[k:, k:])
  #   cc = u * c
  #   A[k:, k:] = A[k:, k:] - 2*cc

  for k in range(n):
    x = A[k:, k]

    e1 = get_e(len(x), 0)
    # e1 = np.zeros(len(x)); e1[0] = 1.0

    u = x + sign(x[0]) * np.linalg.norm(x) * e1
    u = u / np.linalg.norm(u)

    z = np.zeros(n)
    z[k:] = u
    u = z

    P = np.eye(n) - 2 * np.outer(u, u.T)
    A = set_lowVal_zero(P @ A)

    if k < n - 2:
      x = A[k, k+1:]
      e1 = get_e(len(x), 0)
      v = x + sign(x[0]) * np.linalg.norm(x) * e1
      v = v / np.linalg.norm(v)

      z = np.zeros(m)
      z[k+1:] = v
      v = z

      Q = np.eye(m) - 2 * np.outer(v, v.T)
      A = set_lowVal_zero(A @ Q)

  return A

    # A[k:, k:] = A[k:, k:] - 2*u@(u.T*A[k:, k:])

    # a = A[k:m, k:n]
    # x = (u.T * a)
    # xx = a - 2*u*x
    # xxx = a + xx

def set_lowVal_zero(X):
  low_values_indices = abs(X) < 9e-15   # where values are low
  X[low_values_indices] = 0             # all low values set to 0
  return X

def Householder(x, i):
  alpha = -np.sign(x[i]) * np.linalg.norm(x)
  e = np.zeros(len(x)); e[i] = 1.0

  v = (x - alpha * e)
  w = v / np.linalg.norm(v)
  P = np.eye(len(x)) - 2 * np.outer(w, w.T)

  return P

def Golub_Kahan(X):
  col = X.shape[1]
  row = X.shape[0]

  J = X.copy()

  for i in range(col - 2):
    # column
    h = np.zeros(len(J[:, i]))
    h[i:] = J[i:, i]
    P = Householder(h, i)
    J = set_lowVal_zero(P @ J)
    print(J, '\n')

    # row
    h = np.zeros(len(J[i, :]))
    h[i+1:] = J[i, i+1:]
    Q = Householder(h, i+1)
    J = set_lowVal_zero(J @ Q)
    print(J, '\n')

  return J

# A = np.array([[4, 3, 0, 2, 5], [2, 1, 2, 1, 6], [4, 4, 0, 3, 0], [5, 6, 1, 3, 7]])
A = np.array([[4, 3, 0, 2], [2, 1, 2, 1], [4, 4, 0, 3], [5, 6, 1, 3]], dtype='float64')

print(Golub_Kahan(A))
print(gk(A))
pass
