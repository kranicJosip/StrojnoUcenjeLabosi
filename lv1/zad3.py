brojevi = []

while True:
    unos = input("Unesite broj ili 'done' za kraj: ")

    if unos.lower()=="done":
        break

    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Greska molimo unesite broj")

if brojevi:
    print(f"Broj unesenih br je: {len(brojevi)}")
    print(f"Srednja vrijednost je : {sum(brojevi)/len(brojevi)}")
    print(f"Minimalna vrijednost je: {min(brojevi)}")
    print(f"Maksimalna vrijednost je: {max(brojevi)}")
    brojevi.sort()
    print(f"Sortirana lista: {brojevi}")
else:
    print("Niste unjeli niti jedan broj!")
