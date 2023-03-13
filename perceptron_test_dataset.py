import numpy as np
from perceptron import Perceptron  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df=pd.read_csv("iris.csv")
X_data=df.iloc[0:100,[0,2]].values
y_data=df.iloc[0:100,4].values
y_data=np.where(y_data=="setosa",-1,1)


X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.3, random_state=42)


perceptron = Perceptron(dimension = 2, max_iter =100, learning_rate =0.1)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))