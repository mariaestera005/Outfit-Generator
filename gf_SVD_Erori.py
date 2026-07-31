import numpy as np
import matplotlib.pyplot as plt

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

def Afiseaza_Grafic_Convergenta(A_input):

    def SVD_pentru_grafic(A_mat, test_iter):
        m_g, n_g = A_mat.shape

        def Bidiag_Householder(A_h):
            m_b, n_b = A_h.shape
            B_mat = np.copy(A_h).astype(float)
            U_b = np.eye(m_b)
            V_b = np.eye(n_b)

            for k in range(min(m_b, n_b)):
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

                if k < n_b - 2:
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

        _, B_test, _ = Bidiag_Householder(A_mat)
        M_test = B_test.T @ B_test

        def QR_iteration(A_qr, Q_init, MAX_ITER_qr):
            T = Q_init.T @ A_qr @ Q_init
            V = Q_init.copy()

            def Factorizare_QR(A_f):
                m_f, n_f = A_f.shape
                Q_f = A_f.copy().astype(float)
                R_f = np.zeros((n_f, n_f))
                for k in range(n_f):
                    for i in range(k):
                        R_f[i, k] = Q_f[:, i].T @ Q_f[:, k]
                        Q_f[:, k] = Q_f[:, k] - R_f[i, k] * Q_f[:, i]
                    R_f[k, k] = np.linalg.norm(Q_f[:, k])
                    if R_f[k, k] > 1e-12:
                        Q_f[:, k] = Q_f[:, k] / R_f[k, k]
                return Q_f, R_f

            for _ in range(MAX_ITER_qr):
                Q_curent, R_curent = Factorizare_QR(T)
                T = R_curent @ Q_curent
                V = V @ Q_curent
            return T

        Q_init = np.eye(M_test.shape[0])

        T_diag = QR_iteration(M_test, Q_init, test_iter)

        valori_proprii = np.diag(T_diag)
        valori_singulare = np.sqrt(np.maximum(valori_proprii, 0))
        return np.sort(valori_singulare)[::-1]

    _, SR_numpy, _ = np.linalg.svd(A_input, full_matrices=False)

    lista_iteratii = list(range(1, 101, 3))
    lista_erori = []

    for it in lista_iteratii:
        val_algoritm_test = SVD_pentru_grafic(A_input, it)
        eroare = np.max(np.abs(val_algoritm_test - SR_numpy))
        lista_erori.append(eroare)

    plt.figure(figsize=(10, 6))
    plt.semilogy(lista_iteratii, lista_erori, marker='o', color='red', linestyle='-', markersize=4)

    plt.title("Convergența Algoritmului SVD: Evoluția Erorii", fontsize=14, fontweight='bold')
    plt.xlabel("Numărul Maxim de Iterații Alocate", fontsize=12)
    plt.ylabel("Eroare absolută maximă (Scară Logaritmică)", fontsize=12)

    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.show()

Afiseaza_Grafic_Convergenta(A)