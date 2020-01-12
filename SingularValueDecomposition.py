import numpy as np
from math import hypot, sqrt

# Bidiagonalization code based on https://github.com/eidelen/EidLA
class Bidiagonalization(object):
    def householder(self, M):
        m, n = M.shape
        e = np.zeros((m,1)); e[0] = 1.0

        sign = -1.0 if M[0,0] > 0.0 else 1.0

        v = M - (e * np.linalg.norm(M) * sign)
        b = 2.0 / (v.T @ v)[0,0]

        return b, v

    def householder_matrix(self, b, v):
        m, n = v.shape
        p = np.identity(m)

        if not np.isinf(b):
            p = p - b * v @ v.T

        return p

    def bidiagonalize(self, M):
        m, n = M.shape
        if m < n:
            raise Exception('Matrix must have m >= n')

        A = M.copy()
        U = np.identity(m)
        V = np.identity(n)

        for j in range(n):
            x_r = A[j:,j:j+1]
            b, v = self.householder(x_r)
            h_mat_r = self.householder_matrix(b, v)

            a_sub_r = A[j:,j:]

            # tranform with householder matrix
            a_sub_r = h_mat_r @ a_sub_r
            A[j:,j:] = a_sub_r

            # concatenate householder matrix to U
            H_MAT_R = np.identity(m)
            H_MAT_R[j:,j:] = h_mat_r

            U = U @ H_MAT_R

            if j < n - 2:
                x_c = A[j:j+1,j+1:]
                b, v = self.householder(x_c.T)
                h_mat_c = self.householder_matrix(b, v)

                a_sub_c = A[j:,j+1:]

                # tranform with householder matrix
                a_sub_c = a_sub_c @ h_mat_c
                A[j:,j+1:] = a_sub_c

                # concatenate householder matrix to V
                H_MAT_C = np.identity(n)
                H_MAT_C[j+1:,j+1:] = h_mat_c

                V = V @ H_MAT_C

        return U, A, V

class SVD(object):
    def __init__(self, epsilon = np.finfo(np.float).eps):
        super().__init__()

        self.epsilon = epsilon

    def find_q(self, B):
        (m, n) = B.shape
        q = 0

        for i in range(n):
            x = n - 1 - i
            y = n - 1 - i - 1

            if x == 0:
                return q + 1
            if np.abs(B[y, x]) < self.epsilon:
                q += 1
            else:
                return q

    def find_p(self, B, q):
        (m, n) = B.shape
        p = n - q

        for i in range(q, n):
            x = n - 1 - i
            y = n - 1 - i - 1

            if x == 0:
                return p - 1
            if np.abs(B[y, x]) >= self.epsilon:
                p -= 1
            else:
                return p - 1

    def get_closer_number(self, numbers, number):
        diff_a = np.abs(numbers[0] - number)
        diff_b = np.abs(numbers[1] - number)

        if diff_a < diff_b:
            return numbers[0]
        else:
            return numbers[1]

    def get_givens_rotation_matrix(self, a, b):
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

    def svd_internal(self, B):
        (m, n) = B.shape

        T = B.T.dot(B)
        T_trailing = T[m - 2:m,n - 2:n]

        eigenvalues = np.linalg.eig(T_trailing)[0]
        shift = self.get_closer_number(eigenvalues, T_trailing[1, 1])

        y = T[0, 0] - shift
        z = T[0, 1]

        U = np.identity(n)
        V = np.identity(m)

        for k in range(n - 1):
            rotation_matrix = self.get_givens_rotation_matrix(y, z)
            x = B[:, k:k + 2]
            x = x.dot(rotation_matrix)
            B[:, k:k + 2] = x

            V_rotation_matrix = np.identity(n)
            V_rotation_matrix[k:k + 2, k:k + 2] = rotation_matrix
            V = V.dot(V_rotation_matrix)

            y = B[k, k]
            z = B[k + 1, k]

            rotation_matrix = self.get_givens_rotation_matrix(y, z)
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

    def svd(self, M, max_iterations=10000):
        B = M.copy()
        (m, n) = B.shape

        if m != n:
            raise Exception('Dimension must be the same')

        bidiag_U, B, bidiag_V = Bidiagonalization().bidiagonalize(B)

        U = np.identity(m)
        V = np.identity(n)

        q = 0
        iteration = 0
        while q < n and iteration < max_iterations:
            iteration += 1
            q = self.find_q(B)
            p = self.find_p(B, q)

            if q < n:
                b22_size = n - q - p
                b22 = B[p:p + b22_size + 1, p:p + b22_size + 1]

                (u, S, v) = self.svd_internal(b22)

                B[p:p + b22_size + 1, p:p + b22_size + 1] = S

                pu = np.identity(b22_size + p + q)
                pu[p:p + b22_size + 1, p:p + b22_size + 1] = u

                pv = np.identity(b22_size + p + q)
                pv[p:p + b22_size + 1, p:p + b22_size + 1] = v

                U = pu.dot(U)
                V = V.dot(pv)

        U = bidiag_U @ U.T
        V = bidiag_V @ V

        return (U, B, V.T)

M = np.array([1, 2, 1, -1, 1, 2, 4, 2, 2], dtype='float')
M = np.reshape(M, (3,3))

u, b, v = SVD().svd(M, 2000)
