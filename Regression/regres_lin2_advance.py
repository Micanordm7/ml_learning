import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, validation_curve #pour la courbe de validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from pathlib import Path

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
# ---------------------------------------------------------
# 1. ARCHITECTURE DU PIPELINE EXPLORATOIRE
# ---------------------------------------------------------
# On définit explicitement le nom de notre étape de transformation : 'poly'
pipeline_analyse = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('lr', LinearRegression())
])

# ---------------------------------------------------------
# 2. CONFIGURATION ET CALCUL DE LA COURBE DE VALIDATION
# ---------------------------------------------------------
# Nous allons tester de manière empirique les degrés 1, 2, 3 et 4
degres = np.arange(1, 5)

# validation_curve va injecter ces degrés un par un dans l'étape 'poly__degree'
train_scores, val_scores = validation_curve(
    pipeline_analyse, 
    X_train, 
    y_train,
    param_name="poly__degree",  # Ciblage précis du paramètre dans le Pipeline
    param_range=degres,
    cv=5,                       # Validation croisée 5-folds
    scoring="r2"                # Métrique de performance : Coefficient de détermination R²
)

# ---------------------------------------------------------
# 3. CALCUL DES MOYENNES ET ANALYSE
# ---------------------------------------------------------
# La fonction renvoie les scores pour chaque fold, on en fait la moyenne
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

print("--- Résultats de l'analyse de complexité ---")
for d, score_t, score_v in zip(degres, train_mean, val_mean):
    print(f"Degré {d} -> R² Train : {score_t:.4f} | R² Validation : {score_v:.4f}")

# ---------------------------------------------------------
# 4. VISUALISATION GRAPHIQUE
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(degres, train_mean, label="Score Entraînement (Train)", color="blue", marker="o")
plt.plot(degres, val_mean, label="Score Validation (Cross-Val)", color="red", marker="s")
plt.title("Courbe de Validation : Choix de la complexité optimale")
plt.xlabel("Complexité du modèle (Degré Polynomial)")
plt.ylabel("Performance (R2)")
plt.xticks(degres)
plt.ylim(0, 1.1)
plt.legend(loc="best")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 1. ARCHITECTURE DU PIPELINE DE PRODUCTION
# ---------------------------------------------------------
# Le pipeline regroupe nos transformations et l'estimateur final.
# On utilise ici le Degré 2 déterminé comme optimal à l'étape 3.
modele_final = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# ---------------------------------------------------------
# 2. ENTRAÎNEMENT DE L'ENSEMBLE DU PIPELINE
# ---------------------------------------------------------
# La méthode .fit() effectue les actions suivantes en cascade :
#   - poly.fit_transform(X_train)
#   - scaler.fit_transform(...)
#   - regression.fit(...)
modele_final.fit(X_train, y_train)

print("✅ Le pipeline complet a été entraîné avec succès !")
print("Le modèle est maintenant prêt pour l'évaluation finale.")

# ---------------------------------------------------------
# 1. GÉNÉRATION DES PRÉDICTIONS
# ---------------------------------------------------------
# Le pipeline applique automatiquement le scaler de l'entraînement sur X_test
y_pred = modele_final.predict(X_test)

print(f"Nombre de prédictions générées : {y_pred.shape[0]}")

# ---------------------------------------------------------
# 2. CALCUL DES MÉTRIQUES DE PERFORMANCE
# ---------------------------------------------------------
score_r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Alternative universelle à root_mean_squared_error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("\n==================================================")
print("       RAPPORT D'ÉVALUATION DU MODÈLE FINAL       ")
print("==================================================")
print(f"Précision globale (R²)            : {score_r2:.4f}")
print(f"Erreur Moyenne Absolue (MAE)       : {mae:.2f} kg")
print(f"Racine de l'Erreur Quadratique (RMSE) : {rmse:.2f} kg")
print(f"Marge d'erreur relative (MAPE)     : {mape:.2f} %")
print("==================================================")

# ---------------------------------------------------------
# 3. DIAGNOSTIC VISUEL : GRAPHIQUE DES RÉSIDUS
# ---------------------------------------------------------
# Calcul de l'écart entre la réalité et la prédiction
residus = y_test - y_pred

plt.figure(figsize=(8, 5))
# On trace les résidus en fonction des valeurs prédites
plt.scatter(y_pred, residus, alpha=0.6, color='purple', edgecolors='k')

# Ligne d'erreur nulle (le modèle parfait)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label="Erreur Nulle (Idéal)")

plt.title("Graphique de Diagnostic : Analyse des Résidus (Modèle Degré 2)")
plt.xlabel("Poids prédit par le modèle (\hat{y})")
plt.ylabel("Écart avec la réalité (Résidus en kg)")
plt.legend()
plt.grid(True)
plt.show()
