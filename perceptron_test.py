import numpy as np
from perceptron import Perceptron


#creation des donnees apprentissage
#entrees : 4 vecteurs d’entree de dimension 2


X_train = []
X_train.append(np.array([1, 1]))
X_train.append(np.array([1, 0]))
X_train.append(np.array([0, 1]))
X_train.append(np.array([0, 0]))

#sorties : classes correspondant aux 4 entrees
y_train = np.array([1, -1, -1, -1])

#creation du classifieur
perceptron = Perceptron(dimension = 2, max_iter =100, learning_rate =0.1)

#entrainement du classifieur−>calcul des coefficients de l’hyperplan
perceptron.fit(X_train, y_train)

#prediction de nouvelles entrees
new_x = np.array([1, 1])
print(perceptron.predict(new_x))
new_x = np.array([0, 1])
print(perceptron.predict(new_x))