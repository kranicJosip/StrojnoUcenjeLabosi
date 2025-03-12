import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("tiger.png")
if img.dtype == np.uint8:
    img = img.astype(np.float32) / 255.0

bright_img = img + 0.2
bright_img = np.clip(bright_img, 0, 1)

# Kreiranje zrcaljenih slika
# mirrored_img = img[:, ::-1]   Horizontalno zrcaljenje
# flipped_img = img[::-1, :]    Vertikalno zrcaljenje

# rotated_img = np.rot90(img, k=-1)
# ovo je za rotaciju slike

# Smanjenje rezolucije
# low_res_img = img[::x, ::x]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img)
axs[0].set_title("Originalna slika")
axs[0].axis("off")

axs[1].imshow(bright_img)
axs[1].set_title("Posvijetljena slika")
axs[1].axis("off")

plt.show()
