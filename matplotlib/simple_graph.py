import matplotlib.pyplot as plt

#LEs donnees
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]


# La tracée directe
plt.plot(x,y,color='red',marker='*')
plt.title("Style Pyplot (Simple)")
plt.xlabel(" Axe des abcisses")
plt.ylabel("Axe des ordonnées")
plt.show()