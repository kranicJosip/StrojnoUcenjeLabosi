import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
mtcars = pd.read_csv('mtcars.csv')
mpg_by_cyl = mtcars.groupby('cyl')['mpg'].mean()
plt.figure(figsize=(8, 5))
plt.bar(mpg_by_cyl.index.astype(str), mpg_by_cyl.values, color=['blue', 'green', 'red'])
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječna potrošnja (mpg)")
plt.title("Potrošnja automobila prema broju cilindara")
plt.show()

# 2. Boxplot distribucije težine automobila s 4, 6 i 8 cilindara
plt.figure(figsize=(8, 5))
mtcars.boxplot(column='wt', by='cyl', grid=False)
plt.xlabel("Broj cilindara")
plt.ylabel("Težina (1000 lbs)")
plt.title("Distribucija težine automobila prema broju cilindara")
plt.suptitle("") # Uklanjanje automatskog naslova
plt.show()

# 3. Boxplot potrošnje za ručni i automatski mjenjač
plt.figure(figsize=(8, 5))
mtcars.boxplot(column='mpg', by='am', grid=False)
plt.xlabel("Mjenjač (0 = Automatski, 1 = Ručni)")
plt.ylabel("Potrošnja (mpg)")
plt.title("Potrošnja prema tipu mjenjača")
plt.suptitle("")
plt.show()

# 4. Scatter plot odnosa ubrzanja i snage po tipu mjenjača
plt.figure(figsize=(8, 5))
for am_type, group in mtcars.groupby('am'):
    plt.scatter(group['hp'], group['qsec'], label=f"Mjenjač {am_type}")
plt.xlabel("Snaga (HP)")
plt.ylabel("Ubrzanje (sekunde)")
plt.title("Ubrzanje vs Snaga po tipu mjenjača")
plt.legend()
plt.show()