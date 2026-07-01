import matplotlib.pyplot as plt
import numpy as np

# Génération de données aléatoires pour l'exemple
# Classe A : moyenne de 12/20
notes_classe_A = np.random.normal(12, 2, 100) 
# Classe B : moyenne de 14/20
notes_classe_B = np.random.normal(14, 2, 100)

# Définir les mêmes "bins" pour les deux pour une comparaison juste
colonnes = np.linspace(5, 20, 15)

fig, ax = plt.subplots()

ax.hist(notes_classe_A, bins=colonnes, alpha=0.5, color='blue', label='Classe A', edgecolor='black')

# Graphique 2 : Classe B (en orange, superposé)
ax.hist(notes_classe_B, bins=colonnes, alpha=0.5, color='orange', label='Classe B', edgecolor='black')

# Personnalisation
ax.set_title("Comparaison des notes entre deux classes")
ax.set_xlabel("Notes / 20")
ax.set_ylabel("Nombre d'élèves")

plt.legend()
plt.show()