import numpy as np
from perceptron import Perceptron  
# c r e a t i o n des donnees a p p r e n t i s s a g e
# e n t r e e s : 4 v e c t e u r s d ’ en t r e e de dimension 2
X_train = []
X_train.append(np.array([1, 1]))
X_train.append(np.array([1, 0]))
X_train.append(np.array([0, 1]))
X_train.append(np.array([0, 0]))
# s o r t i e s : c l a s s e s co r r espondan t aux 4 e n t r e e s
y_train = np.array([1, -1, -1, -1])
# c r e a t i o n du c l a s s i f i e u r
perceptron = Perceptron(dimension = 2, max_iter =100, learning_rate =0.1)
# en t ra inemen t du c l a s s i f i e u r −> c a l c u l des c o e f f i c i e n t s de l ’ hyp e rp lan
perceptron.fit(X_train, y_train)
# p r e d i c t i o n de n o u v e l l e s e n t r e e s
new_x = np.array([1, 1])
print(perceptron.predict(new_x))
new_x = np.array([0, 1])
print(perceptron.predict(new_x))