# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:23:10 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/11-Linear-Regression")
cwd = os.getcwd()
print(cwd)

#Loading data
df = pd.read_csv("USA_Housing.csv")
df.head()

#Exploring data
df.info()
df.describe() #Summary for numerical variables
df.columns

#Plots
sns.pairplot(df)
sns.distplot(df['Price']) #Normally distributed
sns.heatmap(df.corr(), annot=True)

#Split predictors and target variable
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df["Price"]

#Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#Linear Regression model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
fit = lm.fit(X_train, y_train)

#Model evaluation (train)
print(lm.intercept_)
print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, X.columns, columns = ['Coeff'])

#Model predictions
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions, )

#Model error
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))









