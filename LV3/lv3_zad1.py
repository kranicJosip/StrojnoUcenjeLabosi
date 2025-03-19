import pandas as pd
import numpy as np
import matplotlib as plt
mtcars = pd.read_csv('mtcars.csv')
top5_carsMe = mtcars.sort_values(by=['mpg'])
print(top5_carsMe)
print("---------------------------")
print(top5_carsMe.head(5))
print("---------------------------")
print(top5_carsMe.tail(3))
print("---------------------------")
# 2. Koja tri automobila s 8 cilindara imaju najmanju potrošnju?
cyl8_cars = mtcars[mtcars['cyl'] == 8]
top3_8_cyl = cyl8_cars.sort_values(by=['mpg'])
print(top3_8_cyl.tail(3))
print("---------------------------")
# 3. Srednja potrošnja automobila sa 6 cilindara
cyl6_cars = mtcars[mtcars['cyl'] == 6]
print(cyl6_cars.iloc[:,1].mean())
print("---------------------------")
# 4. Srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs
filtered_4cyl = mtcars[(mtcars['cyl'] == 4) & ((mtcars['wt'] * 1000 >= 2000) & (mtcars['wt'] * 1000 <= 2200))]
avg_mpg_4cyl = filtered_4cyl['mpg'].mean()

# 5. Broj automobila s ručnim i automatskim mjenjačem
manual_count = (mtcars['am'] == 1).sum()
auto_count = (mtcars['am'] == 0).sum()

print( manual_count)
print(auto_count)
print("---------------------------")
# 6. Broj automobila s automatskim mjenjačem i snagom preko 100 KS


count = (mtcars[(mtcars['am']==0) & (mtcars['hp']>100)]).car.count()


print(count)
print("---------------------------")















auto_hp100_count = mtcars[(mtcars['am'] == 0) & (mtcars['hp'] > 100)].shape[0]

# 7. Masa svakog automobila u kilogramima
mtcars['wt_kg'] = mtcars['wt'] * 453.592


