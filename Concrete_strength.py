# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:53:09 2021

@author: Dragonor
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Project - 1 Data (Concrete Data).csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
#X_test[:, 3:] = sc.transform(X_test[:, 3:])
#print(X_train)
#print(X_test)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
DecTree_regressor = DecisionTreeRegressor(random_state = 42)
DecTree_regressor.fit(X, y)


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
ranFor_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
ranFor_regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Predicting the Test set results Decisiojn Tree
y_pred_dec = DecTree_regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_dec.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Predicting the Test set results Random Forest
y_pred_rF = ranFor_regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_rF.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))




#Backward propogation
import statsmodels.api as sm
X = np.append(arr = np.ones((1030, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5,6,7,8]]
regressor_OLS = sm.OLS(endog= y, exog= X_opt).fit()
regressor_OLS.summary()


# Sklearn regression model evaluation functions
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score

#Multiple Linear Regression
# Build some models and check them against training data using MAE, RMSE and R2
lRmodels = [LinearRegression()]
for model in lRmodels:
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_train, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_train, predictions)))
    print("    R2", r2_score(y_train, predictions))
    
    
# Evaluation the models against test data using MAE, RMSE and R2
for model in lRmodels:
    predictions = model.predict(X_test)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_test, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_test, predictions)))
    print("    R2", r2_score(y_test, predictions))


#Decision Tree
# Build some models and check them against training data using MAE, RMSE and R2
dTmodels = [DecisionTreeRegressor(random_state = 42)]
for model in dTmodels:
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_train, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_train, predictions)))
    print("    R2", r2_score(y_train, predictions))
    
    
# Evaluation the models against test data using MAE, RMSE and R2
for model in dTmodels:
    predictions = model.predict(X_test)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_test, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_test, predictions)))
    print("    R2", r2_score(y_test, predictions))
    
    
#Random Forest
# Build some models and check them against training data using MAE, RMSE and R2
rFmodels = [RandomForestRegressor(n_estimators = 10, random_state = 0)]
for model in rFmodels:
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_train, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_train, predictions)))
    print("    R2", r2_score(y_train, predictions))
    
    
# Evaluation the models against test data using MAE, RMSE and R2
for model in rFmodels:
    predictions = model.predict(X_test)
    print(type(model).__name__)
    print("    MAE", mean_absolute_error(y_test, predictions))
    print("    RMSE", sqrt(mean_squared_error(y_test, predictions)))
    print("    R2", r2_score(y_test, predictions))
    

#EVALUATION OF THE Linear Regression MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   


#EVALUATION OF THE Deision Tree Regressor MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_dec)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   

#EVALUATION OF THE Random Forest Regressor MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred_rF)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16) 


