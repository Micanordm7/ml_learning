import matplotlib.pyplot as plt
import numpy as np

# Génération de fausses données (25 tailles autour de 175 cm)
tailles = [165, 172, 168, 180, 175, 188, 171, 192, 160, 174, 
           176, 179, 182, 173, 167, 170, 178, 185, 172, 169, 
           177, 181, 163, 175, 174]


fig, ax = plt.subplots()

# Dessin de l'histogramme
ax.hist(tailles,bins=8,color="red", edgecolor='black')

# Personnalisation
ax.set_title("Distribution des tailles du groupe")
ax.set_xlabel("Taille (en cm)")
ax.set_ylabel("Nombre de personnes")

# Affichage
plt.show()