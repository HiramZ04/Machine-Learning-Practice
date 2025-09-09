from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

#We load the dataset from sklearn 
data = load_iris()

#We call the tran_test_split function on our data to gather a testing and a training X and y variables
xtr, xte, ytr, yte = train_test_split(data.data, data.target)

#We call our KNN Model with neighbors = 1 and fit the training variables
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtr, ytr)

#We do both predictions with training and testing data
pred_tr = knn.predict(xtr)
pred_te = knn.predict(xte)

#We check the shape just for when using column stack and also seeing how much testing and train data we have
print(pred_tr.shape)
print(pred_te.shape)

#We visualize our results with a table having actual data vs predict data from our model
table = np.column_stack((yte[:10],pred_te[:10]))
print(tabulate(table,headers=("Train Data","Test Data"), tablefmt="fancy_grid"))

#We check the score from our knn on the testing data 
print(knn.score(xte,yte))

#We did pretty good!



