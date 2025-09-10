import kagglehub
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("abhishek14398/salary-dataset-simple-linear-regression")

print("Path to dataset files:", path)

#We do an exploratory analysis
df = pd.read_csv("/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv")
df.head(6)

#After checking the data we need to check for NULL values
df.isnull().sum()
#Since there is no NULL values we can proceed with our data cleaning

#First we have to clean some data droping the unnamed column as is an index from the CSV File and we dont need it
df = df.drop(columns=["Unnamed: 0"])
df.head(6)

#We graph our dataset to see if there is an correlation with years of experience and the target variable: Salary
plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Salary on Years Of Experience')
#As we can see there is a STRONG correlation in years of experience and salary outcome so we can start building the model

#We start constructing and training our model
model = LinearRegression()
Data = np.array(df['YearsExperience']).reshape(-1, 1)
Target = np.array(df['Salary'])   
model.fit(Data,Target)
#we store the outpout of the model in a data frame
model_predict = model.predict(Data)

#we do a table of the actual and the predict data
table = np.column_stack((Target,model_predict))

#we show it to see how far our model predicts the results from the actual data
print(tabulate(table[:5], headers =("Actual Data","Predicted Data"), tablefmt = 'fancy_grid'))

#We do the graph again but now with the linear regression plot to visualize it and how well it adheres to the model
plt.scatter(df['YearsExperience'],df['Salary'])
plt.plot(df['YearsExperience'],model_predict,color='r')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Salary on Years Of Experience')
plt.show()

#Then we evaluate our model to see how good we did
#Rsquared to see how good interprets the data
R_squared = np.round(r2_score(Target, model_predict),4)
print("R squared: ",R_squared)

#how much off we are predicting data in dllrs without counting outliers
MAE = np.round(mean_absolute_error(Target,model_predict),4)
print("MAE: ",MAE)


#how much off we are predicting data in dllrs COUNTING outliers
RMSE = np.sqrt(mean_absolute_error(Target,model_predict))
print("RMSE: ",RMSE)

#After interpreting the results we did pretty good, and i mean is a really simple regression so we now start coding for input 
#and output in console from the client 

#We ask the user for an input which would be years of experience and he will expect an output, (salary)
num = int(input("How many years of experience do you have? i will predict your salary:")) 
#We convert the value into an array and reshape it to 2D for the model to accept it 
num = np.array([num]).reshape(-1,1)
#We predict with model using number of user
numpredict = model.predict(num)
 
#We print the number with a short message just to work on conditionals
if numpredict < 30000:
    print("You have to get your experience up!, this is your salary: ", numpredict)
elif numpredict < 100000:
    print("You are getting better but you still have to work on your experience!, this is your salary: ", numpredict)
elif (numpredict >100000):
    print("Wow your experience is great!, this is your salary: ", numpredict)
else:
    print(numpredict)
