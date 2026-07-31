import numpy as np

def SVD_Bidiagonal(A, TOL=1e-6):
    m, n = A.shape

    def Bidiag_Householder(A_mat):
        m_b, n_b = A_mat.shape
        B_mat = np.copy(A_mat).astype(float)
        U_b = np.eye(m_b)
        V_b = np.eye(n_b)

        for k in range(n_b):
            if k < m_b - 1:
                v_stanga = np.copy(B_mat[k:, k])
                v_stanga[0] -= np.sign(v_stanga[0]) * np.linalg.norm(v_stanga)
                if np.linalg.norm(v_stanga) > 1e-14:
                    H_stanga_mic = np.eye(m_b - k) - 2 * np.outer(v_stanga, v_stanga.T) / (v_stanga.T @ v_stanga)
                    H_stanga = np.eye(m_b)
                    H_stanga[k:, k:] = H_stanga_mic
                    B_mat = H_stanga @ B_mat
                    U_b = U_b @ H_stanga

            if k < n_b - 2:
                v_dreapta = np.copy(B_mat[k, k+1:])
                v_dreapta[0] -= np.sign(v_dreapta[0]) * np.linalg.norm(v_dreapta)
                if np.linalg.norm(v_dreapta) > 1e-14:
                    H_dreapta_mic = np.eye(n_b - k - 1) - 2 * np.outer(v_dreapta, v_dreapta.T) / (v_dreapta.T @ v_dreapta)
                    H_dreapta = np.eye(n_b)
                    H_dreapta[k+1:, k+1:] = H_dreapta_mic
                    B_mat = B_mat @ H_dreapta
                    V_b = V_b @ H_dreapta

        return U_b, B_mat, V_b

    U_house, B, V_house = Bidiag_Householder(A)
    M_tridiag = B.T @ B

    def Factorizare_QR(A_mat):
        m_q, n_q = A_mat.shape
        Q_q = A_mat.copy().astype(float)
        R_q = np.zeros((n_q, n_q))
        for k in range(n_q):
            for i in range(k):
                R_q[i, k] = Q_q[:, i].T @ Q_q[:, k]
                Q_q[:, k] = Q_q[:, k] - R_q[i, k] * Q_q[:, i]
            R_q[k, k] = np.linalg.norm(Q_q[:, k])
            if R_q[k, k] > 1e-12:
                Q_q[:, k] = Q_q[:, k] / R_q[k, k]
            else:
                R_q[k, k] = 0.0          # coloana dependenta (rang deficitar) -> NU impartim la 0
                Q_q[:, k] = 0.0
        return Q_q, R_q

    def QR_iteration(M_mat, TOL_QR, max_iter=60):
        n_dim = M_mat.shape[0]
        T_mat = np.copy(M_mat)
        V_mat = np.eye(n_dim)
        for _ in range(max_iter):          # plafon de iteratii -> nu se mai blocheaza la infinit
            off_diagonal_norm = np.linalg.norm(T_mat - np.diag(np.diagonal(T_mat)))
            if off_diagonal_norm < TOL_QR:
                break
            Q_pas, R_pas = Factorizare_QR(T_mat)
            T_mat = R_pas @ Q_pas
            V_mat = V_mat @ Q_pas
        return T_mat, V_mat

    T_diag, V_qr = QR_iteration(M_tridiag, TOL)

    val_proprii = np.diagonal(T_diag)
    val_singulare = np.sqrt(np.maximum(0.0, val_proprii))
    V_complet = V_house @ V_qr

    idx = np.argsort(val_singulare)[::-1]
    val_singulare = val_singulare[idx]
    V = V_complet[:, idx]

    Sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        Sigma[i, i] = val_singulare[i]

    tol_singura = 1e-9
    r = 0
    for i in range(len(val_singulare)):
        if val_singulare[i] > tol_singura:
            r += 1

    U = np.zeros((m, m))
    for i in range(r):
        U[:, i] = (A @ V[:, i]) / val_singulare[i]

    if r < m:
        for i in range(r, m):
            for j in range(m):
                v = np.zeros(m)
                v[j] = 1.0
                for pr in range(i):
                    v -= (U[:, pr].T @ v) * U[:, pr]
                norma = np.linalg.norm(v)
                if norma > 1e-5:
                    U[:, i] = v / norma
                    break

    return U, Sigma, V.T


if __name__ == "__main__":
    A = np.array([[2, 2], [1, 0], [0, 1]]).astype(float)
    U, S, VT = SVD_Bidiagonal(A)
    print("U = \n", np.round(U, 2))
    print("S = \n", np.round(S, 2))
    print("V.T = \n", np.round(VT, 2))
    print("U @ S @ V.T = \n", np.round(U @ S @ VT, 2) + 0.0)

    # verificare rapida fata de numpy (doar valorile singulare)
    print("\nvalori singulare numpy:", np.round(np.linalg.svd(A, compute_uv=False), 4))
