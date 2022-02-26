# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:15:39 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/14-K-Nearest-Neighbors")
cwd = os.getcwd()
print(cwd)

#Loading data
df = pd.read_csv("Classified Data",index_col=0)
df.head()

#Exploring data
df.info()
df.describe() #Summary for numerical variables
df.columns

#Standardizing variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, df["TARGET CLASS"], 
                                                    test_size=0.3, random_state=101)

#KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Prediction
pred = knn.predict(X_test)

#Evaluation
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

#Optimizing number of neighbors (K) using elbow method
import sklearn.metrics

error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test,pred)
    error_rate.append(1-acc)

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle='dashed', 
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
#Optimal k around 17

#New Model
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)

#Prediction
pred = knn.predict(X_test)

#Evaluation
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
















