import matplotlib.pyplot as plt
import numpy as np

# Données fictives : Superficie (X) et Prix d'un appartement (Y)
superficie = np.random.normal(70, 20, 50)
prix = superficie * 3000 + np.random.normal(0, 15000, 50) # Prix lié à la superficie + bruit

fig, ax = plt.subplots()

ax.scatter(superficie,prix,color='red', alpha=0.7,edgecolors='black')

ax.set_title("Relation entre Superficie et Prix")
ax.set_xlabel("Superficie ($m^2$)")
ax.set_ylabel("Prix (€)")

plt.show()