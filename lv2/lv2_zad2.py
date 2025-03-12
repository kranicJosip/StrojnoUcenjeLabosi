import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,4,6),delimiter=",", skiprows=1)

mpg = data[:, 0]  # Potrošnja goriva
hp = data[:, 1]   # Konjske snage
wt = data[:,2]
sizes = wt*50

minimalna = min(data[:,0])
maksimalna = max(data[:,0])
prosjecna = sum(data[:,0])/len(data[:,0])

print("Minimalna potrosnja je: ",minimalna)
print("Maksimalna potrosnja je: ",maksimalna)
print("Prosjenca potrosnja je: ",prosjecna)

plt.scatter(hp, mpg, s=sizes, color='blue', alpha=0.6, edgecolors='black')

# Postavke osi i naslova
plt.xlabel("Konjske snage (hp)")
plt.ylabel("Potrošnja goriva (mpg)")
plt.title("Ovisnost potrošnje goriva o konjskim snagama")

# Dodajemo mrežu radi bolje čitljivosti
plt.grid(True)

# Prikaz grafa
plt.show()
