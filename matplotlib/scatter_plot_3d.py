import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Nécessaire pour activer la 3D

# --- Étape 1 : Préparation des données ---
# Nous générons 200 points aléatoires pour X, Y et Z.
# Les valeurs sont comprises entre 0 et 100 pour correspondre à l'image exemple.
n_points = 200
x = np.random.rand(n_points) * 100
y = np.random.rand(n_points) * 100
z = np.random.rand(n_points) * 100


# --- Étape 2 : Création de la figure et de l'axe 3D ---
fig = plt.figure(figsize=(10, 7)) 
# Création de la toile
ax = fig.add_subplot(111, projection='3d') # Ajout d'un axe avec PROJECTION 3D


# --- Étape 3 : Tracé du Scatter 3D ---

# Plot the 3D line (the 'plot3D' counterpart)
ax.plot3D(x, y, z, color='gray', alpha=0.5, label='Trajectoire 3D')

# Nous traçons les points (x, y, z).
# c=z : La couleur dépend de la valeur Z (hauteur).
# cmap='viridis' : Le dégradé de couleur utilisé (du violet au jaune).
# s=40 : Taille des points.
scatter = ax.scatter3D(x, y, z, c=z, cmap='viridis', s=40, edgecolors='w', linewidth=0.5)

# --- Étape 4 : Personnalisation et Affichage ---
ax.set_title("Nuage de Points 3D (Scatter Plot 3D)", fontsize=14)
ax.set_xlabel("Axe X", fontsize=12)
ax.set_ylabel("Axe Y", fontsize=12)
ax.set_zlabel("Axe Z", fontsize=12)

# Optionnel : Définir les limites pour correspondre parfaitement à l'image
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

#Affichage du graphe
plt.show()