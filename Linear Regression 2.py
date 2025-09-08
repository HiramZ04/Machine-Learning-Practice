from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

#We load the data and convert it as data frames
data = load_diabetes(as_frame=True)
X= pd.DataFrame(data.data)
Y= pd.DataFrame(data.target)

#we print data to make exploratory analysis
print(X.head(5))
print(Y.head(5))

#We graph to see if there is any correlation in bmi and output of target
plt.scatter(X['bmi'],Y)
plt.show()

#we load and fit the dataframes in to the model
model = LinearRegression()
model.fit(X,Y)

#we store the outpout of the model in a data frame
model_predict = pd.DataFrame(model.predict(X))

#we do a table of the actual and the predict data
table = np.column_stack(np.array([Y,model_predict]))

#we print the tabulate of 10 data, with headers and fanctgrid
print(tabulate(table[:10], headers=("Y_actual","Y_predict"), tablefmt = 'fancy_grid'))
