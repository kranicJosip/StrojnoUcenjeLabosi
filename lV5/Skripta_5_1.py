'''
Room occupancy classification 

R.Grbic, 2024.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ucitaj podatke za ucenje
df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# Scatter plot
plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])

plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()

'''
a)na nizim temperaturama i nizem CO2 se pretpostavlja vise da je slobodna soba,
 no pri visim temperaturama i vecem CO2  prevladava da je soba zauzeta.
na visim CO2 imamo vise pretpostavke da je soba zauzeta što je logično jer kad  osobe dišu  ispuštaju CO2.

b) učitani skup sadrži 10129 redova odnosno 10129 podataka.
c)logistička razdioba podataka.
'''