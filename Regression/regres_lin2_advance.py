import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, validation_curve #pour la courbe de validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from pathlib import Path

#importation des fonctions de mmetriques
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error

from mpl_toolkits.mplot3d import Axes3D # Nécessaire pour activer la 3D



# Définir la racine de votre projet (par exemple, le dossier du script actuel)
BASE_DIR = Path(__file__).parent

# Lire un chemin relatif écrit dans un fichier de configuration
chemin_relatif = "data/age_vs_poids_vs_taille_vs_sexe.csv"

data_file = BASE_DIR.parent / chemin_relatif

#Importation du dataset
dataframe = pd.read_csv(data_file)

#Les variables predictives
X = dataframe[["sexe","age","taille"]]

#Le variables Cibles
y = dataframe.poids

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

print(f"Taille d'entraiment:{X_train.shape} et target :{y_train.shape} ")
print(f"Taille de test:{X_test.shape} et target :{y_test.shape} ")
#Instanciation et entrainement du modele
model = LinearRegression()
model.fit(X_train,y_train)

#==========================================================#
#               VERIFICATION DU MODELE                     #
#==========================================================#

#Verification du Score sur les données d'entrainement
train_score = model.score(X_train,y_train)


#Verification du Score sur les données d'entrainement
test_score = model.score(X_test,y_test)

#Affichage des scores

print(f"Score d'entrainement : {train_score}")
print(f"Score de test : {test_score}")

#Affichage des coefficients
coef = model.coef_
print(f" Les coefficients : {coef}")


#===================================================#
#           Les metriques                           #

#Calcul des la prediction
y_pred = model.predict(X_test)

print(f"Prediction :{y_pred.shape}")

#LE Score R2
score_r2 = r2_score(y_test, y_pred)
print(f"Score R2 : {score_r2}")

#FONCTION DE COUT MODEL
print(f"Erreur quadratique moyenne: {mean_squared_error(y_test, y_pred)}")
print(f"Erreur Absolue moyenne: {mean_absolute_error(y_test, y_pred)}")
print(f"Porcentage d'erreur absolu moyenne : {mean_absolute_percentage_error(y_test, y_pred)}")


#============================================================#
# Verification de la complexite du modele                    #

# 1. Génération de la Courbe de Validation (Test de plusieurs degrés)

# On crée un pipeline générique
pipeline_analyse = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('lr', LinearRegression())
])


degres = np.arange(1,4)

#print(pipeline_analyse.get_params().keys())
train_scores, val_scores = validation_curve(
    pipeline_analyse, X_train, y_train,
    param_name="poly__degree",
    param_range=degres,
    cv=5,
    scoring="r2"
)

# Moyenne des scores de validation croisée
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores,axis=1)

# Affichage des scores par degré dans la console
print("--- Analyse de l'impact du Degré Polynomial ---")
for d, t, v in zip(degres, train_mean, val_mean):
    print(f"Degré {d} -> R² Train: {t:.4f} | R² Validation: {v:.4f}")

# Graphique 1 : Courbe de Validation
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(degres, train_mean, label="Score Entraînement (Train)", color="blue", marker="o")
plt.plot(degres, val_mean, label="Score Validation (Cross-Val)", color="red", marker="s")
plt.title("Courbe de Validation : Choix de la complexité")
plt.xlabel("Degré Polynomial")
plt.ylabel("Performance ($R^2$)")
plt.xticks(degres)
#plt.ylim(bottom=0)
plt.legend()
plt.grid(True)


# 2. Graphique 2 : Analyse des Résidus (Erreurs) du modèle actuel
y_pred = model.predict(X_test)
residus = y_test - y_pred

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residus, alpha=0.6, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Analyse des Résidus (Modèle Actuel)")
plt.xlabel("Prédictions ($\hat{y}$)")
plt.ylabel("Résidus (Erreurs)")
plt.grid(True)

plt.tight_layout()
plt.show()