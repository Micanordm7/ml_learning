import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
train_score = model.score(X_test,y_test)

#Affichage des scores

print(f"Score d'entrainement : {train_score}")
print(f"Score de test : {train_score}")

