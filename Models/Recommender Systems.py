# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:11:43 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/19-Recommender-Systems")
cwd = os.getcwd()
print(cwd)

#Loading data

#User Data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
df.head()
df.info()
df.describe() #Summary for numerical variables
df.columns

#Movie Data
movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()
movie_titles.info()
movie_titles.describe() #Summary for numerical variables
movie_titles.columns

#Merging data
df = pd.merge(df, movie_titles, on="item_id")
df.head()

#Data exploration
df.groupby('title')['rating'].mean().sort_values(ascending = False).head()
df.groupby('title')['rating'].count().sort_values(ascending = False).head()

#Creating ratings dataframe
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

#Visualization using ratings dataset
#Number of ratings
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)

#Ratings
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)

sns.jointplot(x='rating', y='num of ratings', data = ratings, alpha = 0.5)

#Recommend similar movies
moviemat = df.pivot_table(index = 'user_id', columns='title', values='rating')
moviemat.head()

ratings.sort_values('num of ratings', ascending=False).head(10)

#Use starwars and liarliar
starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()

liarliar_user_ratings = moviemat['Liar Liar (1997)']

#Checking correlations between ratings of users of the two movies
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

#Cleaning
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.sort_values(by = 'Correlation', ascending=False).head(10)

#Filtering by removing movies with less than 100 ratings
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending=False).head()

#Do the same for liarliar
#Cleaning
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending=False).head()

#Advanced Methods (Collaborative filtering and Content-based recommendations)

#Look at the number of unique users and movies
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))

#Train Test Split
from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

#Memory-Based Collaborative Filtering
#user-item filtering: Users who liked this item also liked
#item-item filtering: Users who are similar to you also liked

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

#Calculate cosine similarity
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

#Predictions
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

#Evaluation
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

#Model-based Collaborative Filtering

#Calculate the sparsity level of MovieLens dataset
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

#SVD (Singular Value Decomposition)
import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))






















































