import numpy as np

M = np.array([1, 2, 1, -1, 1, 2, 4, 2, 2], dtype='float')
M = np.reshape(M, (3,3))

bidiagonal_expected = np.array([-4.24264, 2.68742, 0, 1.11022e-16, 3.08733, -0.637842, 0, 1.11022e-16, -0.916139])
bidiagonal_expected = np.reshape(bidiagonal_expected, (3,3))


def householder(M):
    m, n = M.shape
    e = np.zeros((m,1)); e[0] = 1.0

    if M[0,0] > 0.0:
        sign = -1.0
    else:
        sign = 1.0

    # delete
    # norm = np.linalg.norm(M)
    # prod = (e * np.linalg.norm(M) * sign)
    # prod = np.reshape(prod, M.shape)
    # sub = M - prod

    v = M - (e * np.linalg.norm(M) * sign)
    b = 2.0 / (v.T @ v)[0,0]

    return b, v

def householder_matrix(b, v):
    m, n = v.shape
    p = np.identity(m)

    if not np.isinf(b):
        p = p - b * v @ v.T

    return p

def bidiagonalization(M):
    m, n = M.shape
    if m < n:
        raise Exception()

    A = M.copy()
    U = np.identity(m)
    V = np.identity(n)

    for j in range(n):
        x_r = A[j:,j:j+1]
        b, v = householder(x_r)
        h_mat_r = householder_matrix(b, v)

        a_sub_r = A[j:,j:]

        # tranform with householder matrix
        a_sub_r = h_mat_r @ a_sub_r
        A[j:,j:] = a_sub_r

        # concatenate householder matrix to u
        H_MAT_R = np.identity(m)
        H_MAT_R[j:,j:] = h_mat_r

        U = U @ H_MAT_R

        if j < n - 2:
            x_c = A[j:j+1,j+1:]
            b, v = householder(x_c.T)
            h_mat_c = householder_matrix(b, v)

            a_sub_c = A[j:,j+1:] # tady asi bude chyba
            a_sub_c = a_sub_c @ h_mat_c
            A[j:,j+1:] = a_sub_c # chyba

            H_MAT_C = np.identity(n)
            H_MAT_C[j+1:,j+1:] = h_mat_c # asi chyba

            V = V @ H_MAT_C

    return U, A, V

U, A, V = bidiagonalization(M)

print(U, '\n')
print(A, '\n')
print(V, '\n')
