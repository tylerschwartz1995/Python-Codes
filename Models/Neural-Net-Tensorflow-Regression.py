# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:58:42 2022

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
df = pd.read_csv('DATA/kc_house_data.csv')
df.head()

#Exploring data
df.describe()
df.info()

#Distribution of price of homes
plt.figure(figsize=(12,8))
sns.displot(df['price'])

#Countplot of number of bedrooms
sns.countplot(df['bedrooms'])

#Squarefeet against price
plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='sqft_living', data=df)

#Boxplot of bedrooms and price
sns.boxplot(x='bedrooms', y='price', data = df)

#Geographical Properties
plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data = df)

plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='lat', data = df)

plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data = df, hue='price')

#Seeing most expensive houses
df.sort_values('price', ascending=False).head(20)

#Subsetting Bottom 99%
len(df)*0.01
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]

plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat',
                data = non_top_1_perc, hue = 'price',
                palette='RdYlGn', edgecolor = None, alpha=0.2)

#Waterfront
sns.boxplot(x='waterfront', y='price', data = df)

#Feature engineering

#Dropping id variable
df.drop('id', axis = 1, inplace=True)

#Date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)

sns.boxplot(x='year', y='price', data=df)
sns.boxplot(x='month', y='price', data=df)

df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()

df.drop('date', axis = 1, inplace=True)
df.drop('zipcode', axis = 1, inplace=True)

#Train-test split
X = df.drop('price', axis = 1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

#Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creating model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

#Training model
model.fit(x=X_train, y=y_train.values,
          validation_data = (X_test, y_test.values),
          batch_size=128, epochs=400)

#Plotting loss over epochs
losses = pd.DataFrame(model.history.history)
losses.plot()

#Evaluation on test data
from sklearn.metrics import mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test, predictions)
explained_variance_score(y_test, predictions)

plt.scatter(y_test, predictions) #Our predictions
plt.plot(y_test, y_test, 'r') #Perfect predictions

#Errors
errors = y_test.values.reshape(6480,1)-predictions
sns.histplot(errors)

#Predicting on brand new house
single_house = df.drop('price', axis = 1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1,19))
single_house

model.predict(single_house)
df['price'].iloc[0]
































