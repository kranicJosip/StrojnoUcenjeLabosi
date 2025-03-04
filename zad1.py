def izracunajZaradu(br_sati,satnicaa):
    return br_sati*satnicaa

print('Unesite koliko sati ste odradili: ')
brSati = int(input())
print('Koliko ste placeni po satu?: ')
satnica = float(input())
print(f"Zaradili ste {izracunajZaradu(brSati,satnica)} eura")