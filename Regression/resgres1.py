from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

X = [[0],[1],[2]] 
y = [0,3,6]

#Instanciation du model
mdl = LinearRegression()

#Entrainement du model
mdl.fit(X,y)

#Faire une prediction
newX = [[0.5]]
prediction = mdl.predict(newX)
print(f" la prediction de {newX} est {prediction}")


#La prediction reelle
y_pred = mdl.predict(X)
#==========================
#  LES METRIQUES 
#========================

#
mae = mean_absolute_error(y, y_pred)
print(f";'erreur moyenne est {mae}")




#==========================
#  VISUALISATION 
#========================

#Afficher les graphe

#afficher les point reel
plt.scatter(X,y,label="Donnees reelles")

# Le graphe de la prediction 
plt.plot(X,y_pred,label="")



#Line de Regression
plt.xlabel("Absisse")
plt.ylabel("Ordonnee")
plt.title("Regresion Lineaire")
plt.legend()
plt.show()