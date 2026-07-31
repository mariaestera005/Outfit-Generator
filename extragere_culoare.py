import numpy as np
import matplotlib.pyplot as plt
from SVD_Bidiag import SVD_Bidiagonal


def genereaza_cod_rgb_universal(cale_imagine):
    img = plt.imread(cale_imagine)

    if img.max() > 1.0:
        img = img / 255.0

    if img.shape[2] == 4:
        img = img[:, :, :3]

    h_nou = 40
    l_nou = 40
    h_orig, l_orig, _ = img.shape
    img_mica = np.zeros((h_nou, l_nou, 3))

    for y in range(h_nou):
        for x in range(l_nou):
            idx_y = min(int(y * h_orig / h_nou), h_orig - 1)
            idx_x = min(int(x * l_orig / l_nou), l_orig - 1)
            img_mica[y, x, :] = img[idx_y, idx_x, :]

    colturi = np.array([
        img_mica[0, 0],
        img_mica[0, -1],
        img_mica[-1, 0],
        img_mica[-1, -1]
    ])

    culoare_fundal = np.mean(colturi, axis=0)

    R = img_mica[:, :, 0]
    G = img_mica[:, :, 1]
    B = img_mica[:, :, 2]

    UR, SR, VTR = SVD_Bidiagonal(R)
    UG, SG, VTG = SVD_Bidiagonal(G)
    UB, SB, VTB = SVD_Bidiagonal(B)

    comp1_R = SR[0, 0] * np.abs(np.outer(UR[:, 0], VTR[0, :]))
    comp1_G = SG[0, 0] * np.abs(np.outer(UG[:, 0], VTG[0, :]))
    comp1_B = SB[0, 0] * np.abs(np.outer(UB[:, 0], VTB[0, :]))
    comp2_R = SR[1, 1] * np.abs(np.outer(UR[:, 1], VTR[1, :]))
    comp2_G = SG[1, 1] * np.abs(np.outer(UG[:, 1], VTG[1, :]))
    comp2_B = SB[1, 1] * np.abs(np.outer(UB[:, 1], VTB[1, :]))

    harta_contrast = (
            comp1_R + comp1_G + comp1_B +
            comp2_R + comp2_G + comp2_B
    )

    prag_detectie = 0.20 * np.max(harta_contrast)
    pixeli_haina = []
    for y in range(h_nou):
        for x in range(l_nou):
            pixel = img_mica[y, x, :]
            contrast = harta_contrast[y, x]
            dist_fundal = np.linalg.norm(pixel - culoare_fundal)
            if (
                    contrast > prag_detectie
                    and dist_fundal > 0.10
            ):
                pixeli_haina.append(pixel)

    if len(pixeli_haina) == 0:
        for y in range(h_nou):
            for x in range(l_nou):
                pixel = img_mica[y, x, :]
                dist_fundal = np.linalg.norm(pixel - culoare_fundal)
                if dist_fundal > 0.10:
                    pixeli_haina.append(pixel)

    pixeli_haina = np.array(pixeli_haina)
    luminozitati = np.mean(pixeli_haina, axis=1)
    prag_luciu = np.percentile(luminozitati, 85)
    pixeli_fara_luciu = pixeli_haina[
        luminozitati < prag_luciu
        ]

    if len(pixeli_fara_luciu) < 10:
        pixeli_fara_luciu = pixeli_haina

    rgb_mediu = np.median(pixeli_fara_luciu, axis=0)
    rgb_255 = np.round(rgb_mediu * 255).astype(int)

    return rgb_255, img_mica, harta_contrast


if __name__ == '__main__':
    folder_baza = r".\Imagini"
    lista_imagini = [
        '0.png',
        '1.png',
        '2.png',
        '3.png',
        '4.png',
        '5.png',
        '6.png',
        '7.png',
        '8.png',
        '9.png',
        '10.png',
        '11.png',
        '12.png',
        '13.png',
        '14.png',
        '15.png',
        '16.png',
        '17.png',
        '18.png',
        '19.png',
        '20.png',
        '21.png',
        '22.png',
        '23.png',
        '24.png',
        '25.png',
        '26.png',
        '27.png',
        '28.png',
        '29.png',
        '30.png',
        '31.png',
        '32.png',
        '33.png',
        '34.png',
        '35.png',
        '36.png',
        '37.png',
        '38.png',
        '39.png',
        '40.png',
        '41.png',
        '42.png',
        '43.png',
        '44.png',
        '45.png',
        '46.png',
        '47.png',
        '48.png'
    ]

    for nume_fisier in lista_imagini:
        cale_completa = f"{folder_baza}\\{nume_fisier}"
        try:
            cod_rgb, img_mica, harta_contrast = genereaza_cod_rgb_universal(cale_completa)
            print(f"[SUCCES] {nume_fisier} -> RGB calculat: {cod_rgb}")

            img_originala = plt.imread(cale_completa)

            fig, axe = plt.subplots(1, 3, figsize=(12, 4))

            axe[0].imshow(img_originala)
            axe[0].set_title(f"Original\n{nume_fisier}")
            axe[0].axis('off')

            axe[1].imshow(harta_contrast)
            axe[1].set_title("Harta Contrast\n")
            axe[1].axis('off')

            careu = np.zeros((100, 100, 3))
            careu[:, :, 0] = cod_rgb[0] / 255.0
            careu[:, :, 1] = cod_rgb[1] / 255.0
            careu[:, :, 2] = cod_rgb[2] / 255.0

            axe[2].imshow(careu)
            axe[2].set_title(f"RGB Final\n{cod_rgb}")
            axe[2].axis('off')

            plt.tight_layout()
            plt.show()

        except FileNotFoundError:
            print(f"[EROARE] Nu am găsit fișierul {nume_fisier}")

        except Exception as e:
            print(f"[EROARE] {nume_fisier} -> {e}")

    print("\n Procedeu finalizat")