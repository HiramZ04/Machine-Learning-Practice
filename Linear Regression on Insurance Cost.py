from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#We load the dataset into a dataframe called df
df = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/insurance.csv')
df.head(6)

#We transform from strings to ints ('0 , 1, etc.') the next values
le_sex = LabelEncoder().fit(df['sex'])
sex_new = le_sex.transform(df['sex']) 

le_region =  LabelEncoder().fit(df['region'])
region_new = le_region.transform(df['region'])

le_smoker =  LabelEncoder().fit(df['smoker'])
smoker_new = le_smoker.transform(df['smoker'])

#We drop the old string columns
df = df.drop(columns=(['sex','smoker','region']))

#We add the new encoded variables to replace the old ones (these ones can be used for linear regression)
df['sex'] = sex_new
df['region'] = region_new
df['smoker'] = smoker_new 

df.head(6)

#We graph the variables in function of the target variable to see if there are correlations
for i in df.columns:
    if df[i].name != "charges":
        plt.scatter(df[i],df['charges'],alpha=0.3)
        plt.xlabel(df[i].name)
        plt.ylabel('charges')
        plt.show()

# i really just dont see a strong correlation from any independient variable so first im going to try with age only

#We initialize the X and y variables
X = df['age']
y = df['charges']

#We make the train and test splits
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

#We reshape the X variables since we only have age is a 1D and we need 2D for the LinearRegression function
x_train = np.array(x_train).reshape(-1,1)
x_test = np.array(x_test).reshape(-1,1)

#We load the model and fit it with train data
LR = LinearRegression()
LR.fit(x_train,y_train)

#we make the prediction on the test data
pred = LR.predict(x_test)

#We visualize actual VS predicted data
table = np.column_stack((y_test,pred))
print(tabulate(table[:10], headers = ("Actual Data","Predicted Data"), tablefmt="fancy_grid"))

#We run some metrics since i see bad results on the table
R_squared = np.round(r2_score(y_test,pred),4)
print("R squared: ",R_squared)

MAE = np.round(mean_absolute_error(y_test,pred),4)
print("MAE: ", MAE)

RMSE = np.round(np.sqrt(mean_squared_error(y_test,pred)),4)
print("RMSE: ", RMSE)

#ok we are doing TERRIBLE, lets try to get te R squared up to 0.9
#first lets use the whole dataframe

X = df.drop(columns = ('charges'))

xtr, xte, ytr, yte = train_test_split(X,y)

LR2 = LinearRegression()
LR2.fit(xtr,ytr)
pred2 = LR2.predict(xte)
table2 = np.column_stack((yte,pred2))

print(tabulate(table2[:10],headers=("Actual Data","Predicted Data"), tablefmt="fancy_grid"))

#We run some metrics, i kinda see better predictions now
R_squared = np.round(r2_score(yte,pred2),4)
print("R squared: ",R_squared)

MAE = np.round(mean_absolute_error(yte,pred2),4)
print("MAE: ", MAE)

RMSE = np.round(np.sqrt(mean_squared_error(yte,pred2)),4)
print("RMSE: ", RMSE)

#Ok we are doing quite better but this model still is not reliable since it can predict a value with up to 4104 dllrs in error
#and that is really bad for financial predictions, so we are going to try to optimize the data
sc = StandardScaler()
sc.fit(X)
Xsc = sc.transform(X)

xtr, xte, ytr, yte = train_test_split(Xsc,y)

LR3 = LinearRegression()
LR3.fit(xtr,ytr)
pred3 = LR3.predict(xte)
table3 = np.column_stack((yte,pred3))

print(tabulate(table3[:10],headers=("Actual Data","Predicted Data"), tablefmt="fancy_grid"))

#We run some metrics on the new optimized data
R_squared = np.round(r2_score(yte,pred3),4)
print("R squared: ",R_squared)

MAE = np.round(mean_absolute_error(yte,pred3),4)
print("MAE: ", MAE)

RMSE = np.round(np.sqrt(mean_squared_error(yte,pred3)),4)
print("RMSE: ", RMSE)

#So we got more in the r squared metric but we should try a different model like RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(xtr, ytr)
pred_rf = rf.predict(xte)

print("R²:", r2_score(yte, pred_rf))
