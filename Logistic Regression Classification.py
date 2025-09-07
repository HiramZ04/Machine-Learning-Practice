from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd 

dataframe = load_iris(as_frame=True)
df = dataframe.frame
data = load_iris()

X = data.data
y = data.target

model = LogisticRegression()
model.fit(X,y)

print(df.head(5))
print("Prediction: \n",model.predict(X[:5]))