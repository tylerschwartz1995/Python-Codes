# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:38:24 2022

@author: tyler
"""

#Libraries
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.io as pio
pio.renderers.default='browser'

#Working directory
import os 
os.chdir("C:/Users/tyler/Desktop/Python for Data Science and Machine Learning Bootcamp/09-Geographical-Plotting")

#Graph #1
#Loading data
data = dict(type = "choropleth",
            locations = ["AZ","CA","NY"],
            locationmode = 'USA-states',
            colorscale = 'Jet',
            text = ['Arizona', 'California', 'New York'],
            z = [7.8,4.1,6.2],
            colorbar = {'title':'Unemployment Rate'})


#Simple Plot #1
#Creating layout
layout = dict(geo = {'scope':'usa'})

#Creating choromap and plotting
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)

#Graph #2
#Loading data
df = pd.read_csv("2011_US_AGRI_Exports")
df.head()

#Creating data dictionary
data = dict(type = "choropleth",
            locations = df["code"],
            locationmode = 'USA-states',
            colorscale = 'YlOrRd',
            text = df["text"],
            z = df["total exports"],
            marker = dict(line = dict(color = "rgb(255,255,255)", width = 2)),
            colorbar = {'title':'Millions USD'})

#Creating layout
layout = dict(geo = dict(scope = 'usa', 
                         showlakes = True, 
                         lakecolor = 'rgb(85,173,240)'), 
              title = "2011 US Agriculture Exports by State")

#Creating choromap and plotting
choromap2 = go.Figure(data = [data], layout = layout)
iplot(choromap2)

#Graph 3
#Loading data
df = pd.read_csv("2014_World_GDP")
df.head()

#Creating data dictionary
data = dict(type = "choropleth",
            locations = df["CODE"],
            text = df["COUNTRY"],
            z = df["GDP (BILLIONS)"],
            colorbar = {'title':'GDP in Billions USD'})

#Creating layout
layout = dict(geo = dict(showframe = False,
                         projection = {'type':'equirectangular'}), 
              title = "2014 Global GDP")

#Creating choromap and plotting
choromap3 = go.Figure(data = [data], layout = layout)
iplot(choromap3)

#Graph 4
#Loading data
df = pd.read_csv("2014_World_Power_Consumption")
df.head()

#Creating data dictionary
data = dict(type = "choropleth",
            colorscale = 'Viridis',
            reversescale = True,
            locations = df["Country"],
            locationmode = "country names",
            text = df["Text"],
            z = df["Power Consumption KWH"],
            colorbar = {'title':'KWH'})

#Creating layout
layout = dict(geo = dict(showframe = False,
                         projection = {'type':'mercator'}), 
              title = "2014 Global Power Consumption")

#Creating choromap and plotting
choromap4 = go.Figure(data = [data], layout = layout)
iplot(choromap4)










