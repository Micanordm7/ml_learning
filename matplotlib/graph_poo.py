import matplotlib.pyplot as plt

#Les donnees
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]


#fig = plt.figure()

fig, ax = plt.subplots()

ax.plot(x,y,color='red', linestyle='dotted')
ax.set_title("Style Orienté Objet (Propre)")
ax.set_xlabel(" Axe X")
ax.set_ylabel(" Axe Y")

plt.show()

