import json
import numpy as np
import matplotlib.pyplot as plt
from SVD_Bidiag import SVD_Bidiagonal


nume_fisier = "Articole_vestimentare.json"

with open(nume_fisier, "r", encoding="utf-8") as f:
    date_proiect = json.load(f)

Articole_vestimentare = date_proiect["Articole_vestimentare"]

categorii_posibile = ["Top", "Bottom", "Pantofi", "Geaca"]
anotimpuri_posibile = ["Primavara-Toamna", "Vara", "Iarna"]
stiluri_posibile = ["Casual", "Formal", "Sportiv", "Elegant"]

matr_caracteristici = []

for haina in Articole_vestimentare:
    flags_categorie = [1 if haina["Categorie"] == c else 0 for c in categorii_posibile]
    flags_anotimp = [1 if haina["Anotimp"] == a else 0 for a in anotimpuri_posibile]
    flags_stil = [1 if haina["Stil"] == s else 0 for s in stiluri_posibile]

    rgb_normalizat = [val / 255.0 for val in haina["RGB"]]
    rand_complet = flags_categorie + flags_anotimp + flags_stil + rgb_normalizat
    matr_caracteristici.append(rand_complet)

A = np.array(matr_caracteristici, dtype=float)


U, Sigma, VT = SVD_Bidiagonal(A)

k = 2
U_redus = U[:, :k]

x = U_redus[:, 0]
y = U_redus[:, 1]


plt.figure(figsize=(13, 9))

plt.grid(True, linestyle='--', alpha=0.2, zorder=1)

id = [3, 13, 20, 22, 23, 46]

culori_categorii = {
    "Top": "#FF9999",  # Roz pal
    "Bottom": "#66B2FF",  # Albastru deschis
    "Pantofi": "#99FF99",  # Verde deschis
    "Geaca": "#FFCC99"  # Portocaliu deschis
}


deplasari_personalizate = {
    22: (25, -15),
    23: (-22, 18),
    13: (-25, -15),
    46: (25, -22),
    3: (25, 15),
    20: (-25, -18)
}

categorii_adaugate_in_legenda = set()


for i in range(len(Articole_vestimentare)):
    categorie = Articole_vestimentare[i]["Categorie"]
    stil = Articole_vestimentare[i]["Stil"]
    id_haina = Articole_vestimentare[i]["ID"]
    anotimp = Articole_vestimentare[i]["Anotimp"]

    rgb_reala = [val / 255.0 for val in Articole_vestimentare[i]["RGB"]]
    culoare_punct = culori_categorii.get(categorie, "gray")

    if categorie not in categorii_adaugate_in_legenda:
        plt.scatter(x[i], y[i], color=culoare_punct, edgecolor=rgb_reala, linewidth=3.5, s=200, zorder=3,
                    label=categorie, alpha=0.85)
        categorii_adaugate_in_legenda.add(categorie)
    else:
        plt.scatter(x[i], y[i], color=culoare_punct, edgecolor=rgb_reala, linewidth=3.5, s=200, zorder=3, alpha=0.85)


    if id_haina in id:
        text_caseta = f"ID {id_haina}: {categorie}\n({stil}, {anotimp})"
        offset = deplasari_personalizate.get(id_haina, (0, 15))

        plt.annotate(text_caseta,
                     (x[i], y[i]),
                     xytext=offset,
                     textcoords='offset points',
                     ha='center',
                     va='center',
                     fontsize=9.5,
                     fontweight='bold',
                     color='#2C3E50',
                     bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=rgb_reala, lw=2, alpha=0.95),
                     zorder=4)

plt.title('Proiecția Garderobei în Spațiul Latent SVD', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Componenta SVD 1 (Variația de Sezon)', fontsize=12, labelpad=8)
plt.ylabel('Componenta SVD 2 (Variația de Stil)', fontsize=12, labelpad=8)

plt.xlim(min(x) - 0.03, max(x) + 0.03)
plt.ylim(min(y) - 0.03, max(y) + 0.03)


plt.legend(title="Categorii Articole", fontsize=11, title_fontsize=12, loc="upper left", framealpha=0.95)
plt.savefig('grafic_svd_aerisit_final.png', bbox_inches='tight', dpi=300)

plt.show()