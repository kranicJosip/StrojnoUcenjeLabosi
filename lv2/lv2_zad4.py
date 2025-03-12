import numpy as np
import matplotlib.pyplot as plt

def generate_checkerboard(square_size, rows, cols):
    # Kreiranje crnog (0) i bijelog (255) kvadrata
    black = np.zeros((square_size, square_size), dtype=np.uint8)  # Crni kvadrat
    white = np.ones((square_size, square_size), dtype=np.uint8) * 255  # Bijeli kvadrat

    # Stvaranje jednog reda pomoću hstack (horizontalno slaganje)
    row1 = np.hstack([black, white] * (cols // 2))  # Prvi red (crno-bijelo)
    row2 = np.hstack([white, black] * (cols // 2))  # Drugi red (bijelo-crno)

    # Slaganje redova pomoću vstack (vertikalno slaganje)
    checkerboard = np.vstack([row1, row2] * (rows // 2))

    return checkerboard

# Postavke
square_size = 50  # Veličina kvadrata u pikselima
rows, cols = 6, 6  # Broj kvadrata po visini i širini

# Generiranje šahovnice
img = generate_checkerboard(square_size, rows, cols)

# Prikaz slike
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.axis("off")  # Sakrij osi
plt.show()
