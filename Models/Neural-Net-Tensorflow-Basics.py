# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:14:11 2022

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
df = pd.read_csv("DATA/fake_reg.csv")
df.head()

#Data exploration
sns.pairplot(df)

#Train-test split
from sklearn.model_selection import train_test_split

#Convert pandas to numpy for keras
X = df[['feature1', 'feature2']].values
y = df['price'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Shape of training and test sets
X_train.shape
X_test.shape

#Normalizing/Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train) #Only fit on training set

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

#Model as list of layers
#model = Sequential([
    #Dense(units = 2),
    #Dense(units = 2),
    #Dense(units = 2)
    #])

#Can also add layers one by one doing this:
#model = Sequential()
#model.add(Dense(2))
#model.add(Dense(2))
#model.add(Dense(2))

#Build simple model
model = Sequential()

#Hidden nodes
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))

#Output node
model.add(Dense(1))

#Model compiler
model.compile(optimizer='rmsprop', loss='mse')

#Training model
model.fit(X_train, y_train, epochs=250)

#Evaluation

#Plotting loss as a function of epochs
loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)), y = loss)
plt.title("Training Loss per Epoch");

#Comparing train and test scores
training_score = model.evaluate(X_train, y_train, verbose = 0)
test_score = model.evaluate(X_test, y_test, verbose = 0)
print("Training score: ", training_score)
print("Test score: ", test_score)

#Test predictions
test_predictions = model.predict(X_test)
pred_df = pd.DataFrame(y_test, columns=['Test Y'])
test_predictions = pd.Series(test_predictions.reshape(300,))
pred_df = pd.concat([pred_df, test_predictions], axis = 1)
pred_df.columns = ['Test Y','Model Predictions']

#Evaluating predictions
sns.scatterplot(x='Test Y', y='Model Predictions', data = pred_df)
pred_df['Error'] = pred_df['Test Y']-pred_df['Model Predictions']
sns.displot(pred_df['Error'], bins = 30)

#Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])
mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])

#Predicting on brand new data
new_gem = [[998,1000]]
new_gem = scaler.transform(new_gem)
model.predict(new_gem)

#Saving and loading model
from tensorflow.keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
later_model = load_model('my_model.h5')
later_model.predict(new_gem)
