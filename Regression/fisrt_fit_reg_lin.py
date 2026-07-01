import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
#importing the dataset
df = pd.read_csv("age_vs_poids_vs_taille_vs_sexe.csv")

features = df[['sexe', 'age']]

features1 = df[['sexe', 'age', 'taille']]

target = df.poids

target1 = df.poids

""" Viewing the the content of the dataset 
#showing the data
#print(features)

#showing the data
#print(target)

#description of the dataset
print(df.DESCR)"""

""" Training the model"""

regress = LinearRegression()
regress1 = LinearRegression()
regress.fit(features, target)
regress1.fit(features1, target1)


""" VERIFICATION DU MODELE"""

# Affichage du score
print(f" Score du premier modele: {regress.score(features, target)}")
print()

print(f" Score du secpnd modele : {regress1.score(features, target1)}")
print()

print(f"Coeficent du modele 1 : {regress.coef_}")
print()
print(f"Coeficent du modele 2 : {regress1.coef_}")

""" COST FONCTION"""

target_pred = regress.predict(features)
target1_pred = regress1.predict(features)


print(""" 
      FONCTION DE COUT DU PREMIER MODEL
      """)
print(f"Erreur quadratique moyenne: {mean_squared_error(target, target_pred)}")
print(f"Erreur Absolue moyenne: {mean_absolute_error(target, target_pred)}")
print(f"Porcentage d'erreur absolu moyenne : {mean_absolute_percentage_error(target, target_pred)}")

print(""" 
      FONCTION DE COUT DU SECOND MODEL
      """)
print(f"Erreur quadratique moyenne: {mean_squared_error(target1, target1_pred)}")
print(f"Erreur Absolue moyenne: {mean_absolute_error(target1, target1_pred)}")
print(f"Porcentage d'erreur absolu moyenne : {mean_absolute_percentage_error(target1, target1_pred)}")