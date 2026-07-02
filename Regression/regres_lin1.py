import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
X = dataframe[["age","taille"]]

#Le variables Cibles
y = dataframe.poids

#Instanciation et entrainement du modele
model = LinearRegression()
model.fit(X,y)

testData = pd.DataFrame([[150, 151.15]], columns=["age", "taille"])# [[150,151.15]]

predict = model.predict(testData)

print(f"Le prediction de poids pour l'enfant de: \n L'age : {testData["age"].values[0]} \n la taille : {testData["taille"].values[0]} \n est : {predict[0]:.2f} kg" )




#==========================
#  LES METRIQUES 
#========================
#Afficher la prediction global
y_predict =model.predict(X)

#==========================
#  VISUALISATION 
#========================

#implementer la visualiation 3D
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")

ax.scatter3D(dataframe["age"], dataframe["taille"],y,c=y, cmap='viridis', s=40, edgecolors='w', linewidth=0.5,label="Donnees reelles")

ax.set_xlabel("Age")
ax.set_ylabel('Taille')
ax.set_zlabel('Poids')

plt.title("Regression Lineaire")
plt.show()


