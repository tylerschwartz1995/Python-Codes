# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 16:08:47 2022

@author: tyler
"""

#Import libraries
import pandas as pd
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
pio.renderers.default='browser'
import cufflinks as cf

#Data
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df.head()

df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2.head()

df3 = pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df3.head()

#Using Cufflinks and iplot()

#Scatter
df.iplot(kind = "scatter", x='A',y='B', mode = 'markers', size = 15)

#Bar
df2.iplot(kind = "bar", x='Category', y='Values')
df.sum().iplot(kind="bar")

#Boxplot
df.iplot(kind="box")

#3D Surface plot
df3.iplot(kind='surface',colorscale='rdylbu')

#Histogram
df['A'].iplot(kind="hist", bins=25)
df.iplot(kind="hist", bins=25)

#Spread
df[["A","B"]].iplot(kind="spread")

#Bubble plot
df.iplot(kind="bubble", x="A", y="B", size = "C")

#Scatter matrix
df.scatter_matrix()










