from random import *
import numpy as np

class Perceptron:
    def __init__(self, dimension, max_iter,learning_rate):
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = np.random.random(self.dimension)
        self.w0 = random()


    def fit(self, X, y):
        for i in range(self.max_iter):
            j = randint(0, len(X)-1)
            X_j = X[j] 
            y_j = y[j]

            f = self.predict(X_j)
            y_pred = np.sign(f)
            if y_pred != y_j:
                self.w0 = self.w0 + self.learning_rate * y_j
                self.w = self.w + self.learning_rate * y_j * X_j
    
    
    def predict(self, x):
        f = np.dot(x, self.w) + self.w0
        return np.sign(f)