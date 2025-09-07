from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import pandas as pd

#Construct dataframe for exploratory data analysis 
dataframe = load_diabetes(as_frame=True)
df = dataframe.frame
#we load the data set into data variable
data = load_diabetes()
#We load the variable data target and data to X and y to fit into linear regression model
X = data.data
y = data.target

#We put the linear regression function into a variable called model
model = LinearRegression()
#we fit the data into the model, data and target as X and y
model.fit(X,y)

#We print the first 5 rows of the dataframe to see what the data is about
print(df.head(5))

#We apply the linear regression model to the first 5 rows and print the results
print(model.predict(X[:5]))