# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:47:08 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/17-K-Means-Clustering")
cwd = os.getcwd()
print(cwd)

#Creating data
from sklearn.datasets import make_blobs
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)

#Visualize Data
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')

#Creating clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(data[0])
kmeans.cluster_centers_
kmeans.labels_

#Visualization
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')











