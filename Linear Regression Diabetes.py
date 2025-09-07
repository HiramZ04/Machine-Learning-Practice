import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import pandas as pd


#we load the data set into data variable
data = load_diabetes()
#We load the variable data target and data to X and y to fit into linear regression model
X = data.data
y = data.target

#We create a dataframe only on features of real data excluding target for exploratory analysis
df = pd.DataFrame(data.data, columns=data.feature_names )

#We put the linear regression function into a variable called model
model = LinearRegression()
#we fit the data into the model, data and target as X and y
model.fit(X,y)

#We print the first 5 rows of the dataframe to see what the data is about
print(df.head(5))

#We apply the linear regression model to the first 5 rows and print the results
print(model.predict(X[:5]))


# plot feature vs target
plt.scatter(df['bmi'], y)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('BMI vs Target')
plt.show()