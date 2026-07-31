import json
from extragere_culoare import genereaza_cod_rgb_universal

FOLDER_POZE = r".\Imagini"

with open("Articole_vestimentare.json", encoding="utf-8") as f:
    date = json.load(f)

for art in date["Articole_vestimentare"]:
    cale = f"{FOLDER_POZE}/{art['ID']}.png"
    rgb, _, _ = genereaza_cod_rgb_universal(cale)
    art["RGB"] = [int(c) for c in rgb]
    print(f"{art['ID']:2}  {art['Categorie']:8} -> {art['RGB']}")

with open("Articole_vestimentare.json", "w", encoding="utf-8") as f:
    json.dump(date, f, ensure_ascii=False, indent=2)

print("\nBaza de date a fost actualizată")