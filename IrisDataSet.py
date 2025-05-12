import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Caricamento del dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# Estrazione dei valori minimi e massimi degli assi
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# Creazione della figura
plt.figure(1, figsize=(8, 6))

# Disegno dei punti
colors = ['red', 'green', 'blue']
for i in range(len(labels)):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=colors[i], label=labels[i])
# disegna 3 gruppi di punti colorati, ciascuno in un colore diverso, corrispondente a una classe del dataset Iris

plt.legend()

# Etichettatura degli assi
plt.xlabel('Lunghezza del sepalo')
plt.ylabel('Larghezza del sepalo')

# Limitazione dei valori degli assi
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Aggiunta della taratura sugli assi
plt.xticks(())
plt.yticks(())

# Visualizzazione del grafico 2D
plt.show()
