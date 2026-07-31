import numpy as np


def SVD_Bidiagonal(A, TOL=1e-5):
    m, n = A.shape

    def Bidiag_Householder(A_mat):
        m_b, n_b = A_mat.shape
        B_mat = np.copy(A_mat).astype(float)

        U_b = np.eye(m_b)
        V_b = np.eye(n_b)

        for k in range(min(m_b, n_b)):

            # Householder pe coloane
            if k < m_b - 1:
                v_stanga = np.copy(B_mat[k:, k])
                norma = np.linalg.norm(v_stanga)
                if norma > 1e-14:
                    v_stanga[0] += np.sign(v_stanga[0]) * norma
                    v_stanga = v_stanga / np.linalg.norm(v_stanga)
                    H_mic = np.eye(m_b - k) - 2 * np.outer(v_stanga, v_stanga)
                    H = np.eye(m_b)
                    H[k:, k:] = H_mic
                    B_mat = H @ B_mat
                    U_b = U_b @ H

            # Householder pe linii
            if k < n_b - 1:
                v_dreapta = np.copy(B_mat[k, k + 1:])
                norma = np.linalg.norm(v_dreapta)
                if norma > 1e-14:
                    v_dreapta[0] += np.sign(v_dreapta[0]) * norma
                    v_dreapta = v_dreapta / np.linalg.norm(v_dreapta)
                    H_mic = np.eye(n_b - k - 1) - 2 * np.outer(v_dreapta, v_dreapta)
                    H = np.eye(n_b)
                    H[k + 1:, k + 1:] = H_mic
                    B_mat = B_mat @ H
                    V_b = V_b @ H

        return U_b, B_mat, V_b

    U_house, B, V_house = Bidiag_Householder(A)
    M = B.T @ B

    def Factorizare_QR(A_qr):
        m_qr, n_qr = A_qr.shape
        Q = np.zeros((m_qr, n_qr))
        R = np.zeros((n_qr, n_qr))

        for k in range(n_qr):
            v = A_qr[:, k].copy()
            for j in range(k):
                R[j, k] = Q[:, j].T @ v
                v = v - R[j, k] * Q[:, j]
            R[k, k] = np.linalg.norm(v)
            if R[k, k] > 1e-12:
                Q[:, k] = v / R[k, k]

        return Q, R

    def QR_iteration(A, Q_init, TOL=1e-6, MAX_ITER=100):
        T = Q_init.T @ A @ Q_init
        V = Q_init.copy()
        def Factorizare_QR(A):
            m, n = A.shape
            Q = A.copy().astype(float)
            R = np.zeros((n, n))
            for k in range(n):
                for i in range(k):
                    R[i, k] = Q[:, i].T @ Q[:, k]
                    Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]
                R[k, k] = np.linalg.norm(Q[:, k])
                if R[k, k] > 1e-12:
                    Q[:, k] = Q[:, k] / R[k, k]

            return Q, R

        for _ in range(MAX_ITER):
            off_diag = T - np.diag(np.diag(T))
            if np.linalg.norm(off_diag) < TOL:
                break
            Q, R = Factorizare_QR(T)
            T = R @ Q
            V = V @ Q

        return T, V

    Q_init = np.eye(M.shape[0])
    T_diag, V_qr = QR_iteration(M, Q_init, TOL, 100)

    valori_proprii = np.diag(T_diag)
    valori_singulare = np.sqrt(np.maximum(valori_proprii, 0))
    idx_sort = np.argsort(valori_singulare)[::-1]
    valori_singulare = valori_singulare[idx_sort]

    V = (V_house @ V_qr)[:, idx_sort]

    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = valori_singulare[i]

    U = np.zeros((m, m))
    rang = 0
    for s in valori_singulare:
        if s > 1e-7:
            rang += 1

    for i in range(rang):
        U[:, i] = (A @ V[:, i]) / valori_singulare[i]

    if rang < m:
        for i in range(rang, m):
            for j in range(m):
                v = np.zeros(m)
                v[j] = 1
                for p in range(i):
                    v -= (U[:, p].T @ v) * U[:, p]
                norma = np.linalg.norm(v)
                if norma > 1e-6:
                    U[:, i] = v / norma
                    break

    return U, Sigma, V.T

A = np.array([
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
])

_, SR_algoritm, _ = SVD_Bidiagonal(A)
valori_algoritm = np.diag(SR_algoritm)[np.diag(SR_algoritm) > 1e-5]

_, SR_numpy, _ = np.linalg.svd(A, full_matrices=False)
valori_numpy = SR_numpy[SR_numpy > 1e-5]

print("Algoritm SVD_Bidiag:", valori_algoritm)
print("NumPy SVD:         ", valori_numpy)
print("Eroare maxima:     ", np.max(np.abs(valori_algoritm - valori_numpy)))