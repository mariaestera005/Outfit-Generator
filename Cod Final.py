# Proiect realizat de:
# Chelan Maria-Estera
# Neculae Mara-Daniela
# Sturzoiu Maria-Filofteia
# Grupa 161, Calculatoare și Tehnologia Informației

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from SVD_Bidiag import SVD_Bidiagonal

FOLDER_POZE = r".\Imagini"

def incarca(cale="Articole_vestimentare.json"):
    with open(cale, encoding="utf-8") as f:
        return json.load(f)


def construieste_matrice(articole):
    M = []
    for a in articole:
        rand = [
            1 if a["Categorie"] == "Top" else 0,
            1 if a["Categorie"] == "Bottom" else 0,
            1 if a["Categorie"] == "Pantofi" else 0,
            1 if a["Categorie"] == "Extra" else 0,
            1 if a["Anotimp"] == "Vara" else 0,
            1 if a["Anotimp"] == "Iarna" else 0,
            1 if a["Anotimp"] == "Primavara-Toamna" else 0,
            1 if a["Stil"] == "Casual" else 0,
            1 if a["Stil"] == "Formal" else 0,
            1 if a["Stil"] == "Elegant" else 0,
            1 if a["Stil"] == "Sportiv" else 0,
        ]
        M.append(rand)
    return np.array(M, dtype=float)


def pseudoinversa(X):
    U, S, VT = SVD_Bidiagonal(X)
    m, n = X.shape
    Sp = np.zeros((n, m))
    for i in range(min(m, n)):
        if S[i, i] > 1e-10:
            Sp[i, i] = 1.0 / S[i, i]
    return VT.T @ Sp @ U.T


def scoruri(A, rating):
    rating_np = np.array(rating, dtype=float)
    w = pseudoinversa(A) @ rating_np
    return A @ w


def anotimp_potrivit(temp):
    if temp <= 10:
        return "Iarna"
    if temp <= 19:
        return "Primavara-Toamna"
    return "Vara"


def este_neutra(rgb):
    return max(rgb) - min(rgb) < 50


def canal_dominant(rgb):
    r, g, b = rgb
    if r >= g and r >= b:
        return "R"
    if g >= r and g >= b:
        return "G"
    return "B"


def se_asorteaza(c1, c2):
    if este_neutra(c1) or este_neutra(c2):
        return 1.0
    if canal_dominant(c1) == canal_dominant(c2):
        return 1.0
    return 0.3


def asortare_medie(i, tinuta, articole):
    if len(tinuta) == 0:
        return 1.0
    suma = 0.0
    for j in tinuta.values():
        suma = suma + se_asorteaza(articole[i]["RGB"], articole[j]["RGB"])
    return suma / len(tinuta)


def compune_tinuta(articole, rating, temp, stil_dorit):
    sezon = anotimp_potrivit(temp)
    scor = scoruri(construieste_matrice(articole), rating)

    tinuta = {}

    for categorie in ["Top", "Bottom", "Extra", "Pantofi"]:

        articole_potrivite = []

        for i in range(len(articole)):
            if (
                    articole[i]["Anotimp"] == sezon
                    and articole[i]["Categorie"] == categorie
            ):
                articole_potrivite.append(i)

        if len(articole_potrivite) == 0:
            continue

        articole_stil = []

        for i in articole_potrivite:
            if articole[i]["Stil"] == stil_dorit:
                articole_stil.append(i)

        if len(articole_stil) == 0:
            continue

        articol_ales = articole_stil[0]

        for i in articole_stil:

            asortare_noua = asortare_medie(i, tinuta, articole)

            asortare_curenta = asortare_medie(
                articol_ales,
                tinuta,
                articole
            )

            if (
                    asortare_noua > asortare_curenta
                    or (
                    asortare_noua == asortare_curenta
                    and scor[i] > scor[articol_ales]
            )
            ):
                articol_ales = i

        tinuta[categorie] = articol_ales

    return tinuta, sezon


def arata_poze(articole, tinuta, temp, sezon, stil):
    if len(tinuta) == 0:
        print(f"(Nu ai nicio piesa {stil} potrivita pentru {temp:.0f} grade)")
        return

    categorii = list(tinuta.keys())
    n = len(categorii)
    fig, axe = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axe = [axe]

    titlu = f"{temp:.0f} grade ({sezon}) - stil {stil}"
    fig.suptitle(titlu)

    for k in range(n):
        categorie = categorii[k]
        i = tinuta[categorie]
        art = articole[i]

        cale_poza = os.path.join(FOLDER_POZE, str(art["ID"]) + ".png")
        img = plt.imread(cale_poza)
        axe[k].imshow(img)
        axe[k].axis("off")

    plt.show()


if __name__ == "__main__":
    date = incarca()
    articole = date["Articole_vestimentare"]
    rating = date["Ratingul_Utilizatorului"]

    temp = float(input("Ce temperatura e afara (grade C)? "))
    stil = input("Ce stil vrei azi (Casual / Formal / Elegant / Sportiv)? ")
    stil = stil.strip()
    stil = stil.capitalize()

    tinuta, sezon = compune_tinuta(articole, rating, temp, stil)

    print(f"\nPentru {temp:.0f} grade ({sezon}), stil {stil}:")
    for categorie in tinuta:
        i = tinuta[categorie]
        art = articole[i]
        print(f"  {categorie:8} #{art['ID']:2}  {art['Stil']:8}  RGB {art['RGB']}")

    arata_poze(articole, tinuta, temp, sezon, stil)