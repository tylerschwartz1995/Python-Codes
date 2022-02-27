# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 11:28:26 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/16-Support-Vector-Machines")
cwd = os.getcwd()
print(cwd)

#Loading data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
cancer['feature_names']

#Features
X = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
X.head()

#Exploring data
X.info()
X.describe() #Summary for numerical variables
X.columns

#Target
y = pd.DataFrame(cancer["target"], columns = ['Cancer'])

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Training Model
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

#Predictions
predictions= model.predict(X_test)

#Evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Gridsearch
param_grid = {'C': [0.1,1,10,100,1000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
model_grid = GridSearchCV(SVC(), param_grid, refit=True,verbose=3)
model_grid.fit(X_train, y_train)
print(model_grid.best_params_) #Optimized parameters
print(model_grid.best_estimator_)

#Making predictions with optimized hyper-parameters
grid_predictions = model_grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))




































