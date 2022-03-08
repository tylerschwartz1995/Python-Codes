# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:46:44 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/22-Deep Learning")
cwd = os.getcwd()
print(cwd)

#Importing data
df = pd.read_csv('DATA/cancer_classification.csv')
df.head()

#Exploring data
df.describe()
df.info()

#Visualizations
sns.countplot(x='benign_0__mal_1', data = df)

sns.heatmap(df.corr(), cmap='coolwarm')
df.corr()['benign_0__mal_1'].sort_values()
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

#Train-Test Split
X = df.drop("benign_0__mal_1", axis = 1).values
y = df["benign_0__mal_1"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

#Scalling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creating model #1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

model = Sequential()

#Hidden layers
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))

#Output layer
model.add(Dense(units=1, activation='sigmoid'))

#Compile
model.compile(loss = 'binary_crossentropy', optimizer='adam')

#Training model
model.fit(x=X_train, y=y_train,
          epochs = 600,
          validation_data = (X_test, y_test),
          verbose = 1)

#Plotting loss over epochs
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

#Creating model #2 (With Early Stopping to Avoid Overfitting)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

#Hidden layers
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))

#Output layer
model.add(Dense(units=1, activation='sigmoid'))

#Compile
model.compile(loss = 'binary_crossentropy', optimizer='adam')

#Early stopping
early_stop = EarlyStopping(monitor='val_loss',
                           mode = 'min',
                           verbose = 1, patience=25)

#Training model
model.fit(x=X_train, y=y_train,
          epochs = 600,
          validation_data = (X_test, y_test),
          callbacks = [early_stop],
          verbose = 1)

#Plotting loss over epochs
model_loss = pd.DataFrame(model.history.history)
model_loss.plot() #Better

#Creating model #3 (With Dropout layers which also prevents overfitting)
from tensorflow.keras.layers import Dropout

model = Sequential()

#Hidden layers
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

#Output layer
model.add(Dense(units=1, activation='sigmoid'))

#Compile
model.compile(loss = 'binary_crossentropy', optimizer='adam')

#Training model
model.fit(x=X_train, y=y_train,
          epochs = 600,
          validation_data = (X_test, y_test),
          callbacks = [early_stop],
          verbose = 1)

#Plotting loss over epochs
model_loss = pd.DataFrame(model.history.history)
model_loss.plot() 

#Model evaluation
predictions = model.predict_classes(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
















