# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 14:36:17 2022

@author: tyler
"""

#Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#Import data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#Explore
df.head()
df.info()

#Scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

#Comparing shapes
print(scaled_data.shape)
print(x_pca.shape)

#Visualizing PCA components
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'], cmap= "plasma")
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

#Interpreting components
pca.components_
df_comp = pd.DataFrame(pca.components_, columns = cancer["feature_names"])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap= "plasma")


















