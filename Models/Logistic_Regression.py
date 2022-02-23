# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:00:08 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Interactive plot libraries
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
pio.renderers.default='browser'
import cufflinks as cf

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/13-Logistic-Regression")
cwd = os.getcwd()
print(cwd)

#Loading data
train = pd.read_csv("titanic_train.csv")
train.head()

#Exploring data
train.info()
train.describe() #Summary for numerical variables
train.columns

#Heatmap of missing data
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')

#Count plot of target
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue = 'Sex' ,data = train)
sns.countplot(x='Survived',hue = 'Pclass' ,data = train)

#Distribution of age
sns.displot(train['Age'].dropna(), kde = False, bins=30)

#Count plot of sibblings on board
sns.countplot(x='SibSp', data = train)

#Distribution of fare paid
train['Fare'].hist(bins = 30, figsize=(10,4))
#train['Fare'].iplot(kind='hist', bins=50)

#Age by class
plt.figure(figsize= (10,7))
sns.boxplot(x='Pclass', y = 'Age', data = train)

#Imputation for missing values
#Age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age

train["Age"] = train[["Age","Pclass"]].apply(impute_age, axis = 1)

#Heatmap of missing data
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')

#Cabin
train.drop('Cabin', axis = 1, inplace=True)
train.head()

#Drop remaining missing values
train.dropna(inplace=True)

#Converting categorical variables to dummy variables
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#Add dummy columns to dataframe
train = pd.concat([train,sex,embark], axis=1)
train.head()

#Dropping uneeded columns
train.drop(['Sex',"Embarked","Name","Ticket","PassengerId"], axis = 1, inplace=True)
train.info()

#Logistic regression model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis = 1),
                                                    train['Survived'], test_size=0.3,
                                                    random_state = 101)

#Training model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#Model prediction on test set
predictions = logmodel.predict(X_test)

#Model evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))












