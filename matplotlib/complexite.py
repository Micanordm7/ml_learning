import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Génération de données non-linéaires (courbe sinusoïdale + bruit)
np.random.seed(0)
X = np.sort(np.random.rand(30, 1) * 5, axis=0)
y = np.sin(X).ravel() + np.random.randn(30) * 0.15

# Points pour afficher les courbes de prédiction fluides
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]

# Degrés de complexité à tester
degres = [1, 3, 15]
titres = ["Sous-apprentissage (Degré 1)", "Modèle Optimal (Degré 3)", "Surapprentissage (Degré 15)"]
couleurs = ["orange", "green", "red"]

plt.figure(figsize=(15, 5))

for i, degre in enumerate(degres):
    ax = plt.subplot(1, 3, i + 1)
    
    # Création d'un pipeline : Transformation polynomiale -> Régression Linéaire
    model = make_pipeline(PolynomialFeatures(degre), LinearRegression())
    model.fit(X, y)
    
    # Prédiction
    y_plot = model.predict(X_plot)
    
    # Graphique
    plt.scatter(X, y, color="blue", s=30, label="Données d'entraînement")
    plt.plot(X_plot, y_plot, color=couleurs[i], linewidth=2, label=f"Modèle")
    plt.xlim(0, 5)
    plt.ylim(-2, 2)
    plt.title(titres[i])
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.grid(True)

plt.tight_layout()
plt.show()