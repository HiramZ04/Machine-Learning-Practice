from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

data = load_diabetes(as_frame=True)
X= pd.DataFrame(data.data)
Y= pd.DataFrame(data.target)


print(X.head(5))
print(Y.head(5))

plt.scatter(X['bmi'],Y)
#plt.show()

model = LinearRegression()
model.fit(X,Y)

model_predict = pd.DataFrame(model.predict(X))

iii = np.column_stack(np.array([Y,model_predict]))


print(tabulate(iii[:10], headers=("Y_actual","Y_predict"), tablefmt = 'fancy_grid'))
