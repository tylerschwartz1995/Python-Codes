# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:51:55 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/15-Decision-Trees-and-Random-Forests")
cwd = os.getcwd()
print(cwd)

#Loading data
df = pd.read_csv("kyphosis.csv")
df.head()

#Exploring data
df.info()
df.describe() #Summary for numerical variables
df.columns

#Converting target variable to dummy 'Present'
Present = pd.get_dummies(df['Kyphosis'],drop_first=True)

#Add dummy columns to dataframe
df = pd.concat([df.drop('Kyphosis', axis = 1),Present], axis=1)
df.head()

#Pairplot
sns.pairplot(df, hue="present", palette='Set1')

#Train test split
from sklearn.model_selection import train_test_split
X = df.drop('present',axis=1)
y = df['present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

#Prediction
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

#Tree Visualization #1
from sklearn import tree
features = list(df.columns[:-1])
target = list(df.columns[3:])

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree, 
                   feature_names=features,            
                   filled=True)


#Random Forests
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

#Prediction
rfc_pred = rfc.predict(X_test)

#Evaluation
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

#Variable Importance (Permutation)
from sklearn.inspection import permutation_importance
result = permutation_importance(
    rfc, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
forest_importances = pd.Series(result.importances_mean, index=features)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()







