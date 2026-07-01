import matplotlib.pyplot as plt

# Données
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# On crée un seul graphique (un seul 'ax'
fig, ax = plt.subplots(figsize=(6,4))

# 1. On trace la courbe (ligne bleue)
ax.plot(x,y,color='blue',label="Ligne de tendance")

# 2. On superpose les points (ronds rouges)
ax.scatter(x,y,s=100,zorder=3,label="Données réelles") # Note : 'zorder=3' force les points à être dessinés AU-DESSUS de la ligne

# Personnalisation
ax.set_title("Courbe et Points superposés")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend() # Affiche la légende en haut à gauche

plt.show()

