# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:09:02 2022

@author: tyler
"""



import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Importing data
tips = sns.load_dataset('tips')
tips.head()
flights = sns.load_dataset("flights")
flights.head()
iris = sns.load_dataset("iris")
iris.head()

#Setting style
sns.set_style("ticks")

#Despine (remove top, left, right or bottom axis)
#sns.despine()

#Figure size
#plt.figure(figsize=(12,3))

#Size style (i.e poster size)
#sns.set_context("poster")
#sns.set_context("notebook")

#Distribution plot
sns.displot(tips["total_bill"], kde=True, bins=30)

#Joint plot
sns.jointplot(x="total_bill" , y="tip" , data = tips, kind = "hex")

#Pairplot
sns.pairplot(data = tips, hue='sex')

#Rugplot
sns.rugplot(tips['total_bill'])

#Barplot
sns.barplot(x="sex",y="total_bill",data=tips, estimator=np.std)

#Count plot
sns.countplot(x='sex', data=tips)

#Boxplot
sns.boxplot(x="day",y="total_bill", data=tips, hue="smoker")

#Violin plot
sns.violinplot(x="day", y="total_bill", data=tips, hue="sex", split=True)

#Strip plot
sns.stripplot(x="day",y="total_bill", data=tips, jitter=True, hue="sex")

#Swarm plot
sns.swarmplot(x="day",y="total_bill", data=tips)  

#Factorplot
sns.factorplot(x="day",y="total_bill",data=tips,kind='bar')

#Heat map(data needs to be in matrix form)
tc = tips.corr() #matrix form
sns.heatmap(tc, annot=True, cmap="coolwarm")

fp = flights.pivot_table(index="month", columns="year", values="passengers")
sns.heatmap(fp, cmap="magma")

#Clustermap
sns.clustermap(fp, cmap="coolwarm")

#Pairgrid
g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)

#Facet grid
g = sns.FacetGrid(data=tips, col='time', row='smoker')
g.map(plt.scatter,"total_bill", 'tip')

#LM Plot
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex',
           markers=['o','v'])

sns.lmplot(x='total_bill', y='tip', data=tips, col='sex', row= 'time')









