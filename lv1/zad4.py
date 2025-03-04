try:
 
    ime_datoteke = input("Unesite ime datoteke: ")


    with open(ime_datoteke, "r", encoding="utf-8") as file:
        ukupna_pouzd = 0 
        brojac = 0 

        for linija in file:

            if linija.startswith("X-DSPAM-Confidence:"):
                try:
 
                    vrijednost = float(linija.split(":")[1].strip())
                    ukupna_pouzd += vrijednost
                    brojac += 1
                except ValueError:
                    print("Greška pri pretvaranju vrijednosti.")


        if brojac > 0:
            print(f"Average X-DSPAM-Confidence: {ukupna_pouzd / brojac}")
        else:
            print("Nema pronađenih podataka.")

    #except FileNotFoundError:
       # print("Greška: Datoteka nije pronađena.")
except Exception as e:
    print(f"Došlo je do greške: {e}")