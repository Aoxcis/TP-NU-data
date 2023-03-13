import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv("iris.csv")

#recuperation des 100 premieres lignes et des colonnes 0 et 2
#du tableau correspondant aux caracteristiques sepal_length et
#petal_length(entrees)
X_data=df.iloc[0:100,[0,2]].values
#recuperation des 100 premieres lignes de la derniere colonne
#correspondant a l’espece de l’iris (sorties/classes)
y_data=df.iloc[0:100,4].values
#classification binaire : transformation des labels en valeurs −1 ou 1
#setosa <=> −1 ou non setosa <=> 1
y_data=np.where(y_data=="setosa",-1,1)
#attributiondecouleursdifferentesaux2classes
colors={-1:'red',1:'blue'}
y_colors=[colors[y] for y in y_data]
#affichage des nuages de points avec leur classe
plt.scatter(X_data[:,0],X_data[:,1],c=y_colors,s=100)
plt.show()