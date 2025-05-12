from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# CLASSIFIERS PARAMETERS
NUMBER_NEIGHBORS = 5        # value of K in K-NN classifier
HIDDEN_LAYERS = 16          # number of neurons in the hidden layer of the MLP
MAX_ITERATION = 3000        # maximum number of iterations in the MLP

# Caricamento del dataset
iris = load_iris()
X = iris.data[:, :4]  # Tutte e 4 le features
y = iris.target

# Split del dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,
    train_size=0.8, shuffle=True, random_state=0)

# Inizializzazione
# model = KNeighborsClassifier(n_neighbors=1)
model = KNeighborsClassifier(n_neighbors=5)
# model = MLPClassifier(hidden_layer_sizes=(HIDDEN_LAYERS,), max_iter=MAX_ITERATION, verbose=True)

# Addestramento su training set
model.fit(X_train, y_train)

# Valutazione su test set
y_pred = model.predict(X_test)
print("\n%d classificazioni corrette su %d campioni"
      % ((y_test == y_pred).sum(), X_test.shape[0]))
accuracy = (y_test == y_pred).mean()
print(f"Accuratezza: {accuracy:.2%}")



# FASE 1: ADDESTRAMENTO
# - di caricamento del dataset
# - training set
# - test set

# FASE 2: VALUTAZIONE DEL MODELLO 
# - 2 print con accuratezza

# FASE 3: FASE OPERATIVA
# - predict per determinare la classe di appartenenza





