import matplotlib.pyplot as plt

#dreation de la grille
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))

#Premier Graphe
ax1.plot([1,2,3],[4,5,6])
ax1.set_title("Graphique 1")

#deuxieme Graphe
ax2.plot(['A','B','C'],[3,7,5])
ax2.set_title("Graphique 2")

plt.tight_layout()
plt.show()