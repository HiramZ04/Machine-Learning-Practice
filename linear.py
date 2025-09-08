from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# We construct the X array which would be hours of studying
X = np.array([2,2,2.2,2.2,2,2.3,1.5,1,1,2,2.1,1,5,5,4.5,5.5,5,5.2,5.1,6,6,7,7.6,7.3,9,9,8,10,10], dtype=float)
# We construct the y array which would be test grades
y = np.array([70,70,70,70,70,70,60,60,65,70,73,43,80,80,80,80,80,80,80,85,85,90,90,90,100,100,100,100,100], dtype=float)

# X should be 2D 
X = X.reshape(-1, 1)           

# We import and fit the linear regression model with our data
model = LinearRegression()
model.fit(X,y)
predict = model.predict(X)

#We than print in a good format, the actual data vs the predict data to see how far of we are 
table = np.column_stack([y,predict])
print(tabulate(table, headers = ('y_Actual','y_Predict'), tablefmt='fancy_grid'))

#We print a graph to see how the linear regression adheres with the actual data and see visualy if its a good solution
plt.scatter(X,y)
plt.plot(X,predict,'-')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted Data VS Actual Data")
#plt.show()

#After analysis and deciding a linear regression is a good way to predict y since hours of study are correlated directly
# to the greades on the test we  need to then evaluate the accuracy of the model

R_squared = np.round(r2_score(y, predict),4)
print("R-Squared: ",R_squared)

MSE = np.round(mean_squared_error(y, predict),4)
print("MSE: ", MSE)

MAE = np.round(mean_absolute_error(y,predict),4)
print("MAE: ",MAE)

MAPE = np.mean(np.abs((y - predict) / y)) * 100
print(f"MAPE: {MAPE:.2f}%")

RMSE = np.sqrt(MSE)
print("RMSE: ",RMSE)

#After evaluating the model and understanding we can keep going with a UI solution for a user
#We have to develop somewhere a student inputs the hours of study and then the program would predict and output the test grade
num = float(input("How many hours did you study for the exam? "))
num = np.array([num])   
new = np.round(model.predict([num]),2)
print("This is your predicted score for the test based on your hours of study:\n",new)