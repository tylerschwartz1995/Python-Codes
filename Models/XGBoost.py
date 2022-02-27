# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:12:09 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Codes/Sample Data")
cwd = os.getcwd()
print(cwd)

#Loading data
df = pd.read_csv("StreamPromo.csv")
df.head()

#Exploring data
df.info()
df_desc = df.describe() #Summary for numerical variables
df.columns

#Dropping train and test columns
df.drop(['test','train','amount'], inplace=True, axis = 1)

#Visualizations
sns.countplot(x='sale',hue = 'female' ,data = df)
sns.countplot(x='sale',hue = 'spouse' ,data = df)
sns.displot(df["amountlast"], kde=True, bins=30)
sns.jointplot(x="amountlast" , y="amountyear" , data = df, hue = "sale")
sns.boxplot(x="female",y="salesyear", data=df, hue="sale")

df_corr = df.corr() #matrix form
sns.heatmap(df_corr, cmap="coolwarm")

#Create dummies for categorica variables (revenue and region)
revenue = pd.get_dummies(df['revenue'],drop_first=True)
region = pd.get_dummies(df['region'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df,revenue], axis=1)
df.rename(columns={2: 'rev_2', 3: 'rev_3'}, inplace=True)
df = pd.concat([df,region], axis=1)
df.rename(columns={2: 'region_2', 3: 'region_3', 4: 'region_4', 5: 'region_5'}, 
          inplace=True)

#Dropping uneeded columns and renaming columns
df.drop(['region',"revenue"], axis = 1, inplace=True)
df.info()

#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('sale', axis = 1),
                                                    df['sale'], test_size=0.3,
                                                    random_state = 101)

#Model 1: Logistic Regression

#Training model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#Model prediction on test set
predictions = logmodel.predict(X_test)

#Model evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

#Model 2: XGBOOST
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)

#Setting parameters
#Booster parameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

#Training model
num_round = 50
bst = xgb.train(param, dtrain, num_round)

#Prediction
dtest = xgb.DMatrix(X_test)
ypred = bst.predict(dtest)

#Cutoff
for i in range(0,len(ypred)):
    if ypred[i] >= 0.5:
        ypred[i] = 1
    else:
        ypred[i] = 0


#Result Plots
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)

#Evaluation
print(classification_report(y_test, ypred))




















