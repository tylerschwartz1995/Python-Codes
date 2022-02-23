# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:08:41 2022

@author: tyler
"""

import numpy as np
import pandas as pd
import seaborn as sns

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/07-Pandas-Built-in-Data-Viz")
cwd = os.getcwd()
print(cwd)

#Loading data
df1 = pd.read_csv("df1", index_col=0)
df1.head()

df2 = pd.read_csv("df2", index_col=0)
df2.head()

df = pd.DataFrame(np.random.randn(1000,2), columns = ['a', 'b'])
df.head()

#Displays

#Histogram
df1['A'].hist(bins=30)
df1['A'].plot(kind= "hist", bins= 30)
df1['A'].plot.hist()

#Area plot
df2.plot.area(alpha = 0.4)

#Barplot
df2.plot.bar(stacked = True)

#Line plot
df1.plot.line(y ='B', figsize = (12,3))

#Scatter plots
df1.plot.scatter(x='A',y='B', c = 'C')

#Boxplot
df1.plot.box()

#Hex Plot (Bivariate)
df.plot.hexbin(x='a',y='b', gridsize=25)

#Kernal density plot
df['a'].plot.kde()
df2.plot.density()






















