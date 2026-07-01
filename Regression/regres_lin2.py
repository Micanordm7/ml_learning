import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Importation du dataset
datafile = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")

#Les variables predictives
X = datafile[["age","taille"]]

#Le variables Cibles
y = datafile.poids

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

ax.scatter(datafile["age"], datafile["taille"],y,color="red",label="Donnees reelles")

ax.set_xlabel("Age")
ax.set_ylabel('Taille')
ax.set_zlabel('Poids')

plt.title("Regression Lineaire")
plt.show()


