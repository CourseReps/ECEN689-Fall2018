#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:05:15 2018

@author: harinath
"""

import pandas as pd
#from google.colab import files 
import numpy as np


trainData = '/Users/harinath/Desktop/ECEN689/Health.csv'
health_df = pd.read_csv(trainData)

#hud_df = pd.read_csv(io.StringIO(uploaded_hud['COUNTY_ZIP_122016.csv'].decode('cp1252')))
health_df.head(10)

trainDataSource = '/Users/harinath/Downloads/FIPSFilteredData.csv'
income_df = pd.read_csv(trainDataSource)
income_df.head(10)

combined= pd.merge(health_df, income_df, on='FIPS')
print(combined.shape)

import statsmodels.api as sm # import statsmodels 

X = combined["PCT_DIABETES_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined["A02650"] ## Y usually means our output/dependent variable
#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print(model.summary())

from scipy.stats import linregress
print(linregress(X, y))

X = combined["PCT_DIABETES_ADULTS13"] ## X usually means our input variables (or independent variables)
y = combined["A02650"]
print(linregress(X, y))

X = combined["PCT_OBESE_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined["A02650"]
print(linregress(X, y))

X = combined["PCT_OBESE_ADULTS13"] ## X usually means our input variables (or independent variables)
y = combined["A02650"]
print(linregress(X, y))

combined.to_csv('Health_IRS.csv')


import plotly as py
py.tools.set_credentials_file(username='HariNath16d5', api_key='dMlY7TFU46AM1t5I8cFL')
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df['text'] = df['state'] + '<br>' +\
    'Beef '+df['beef']+' Dairy '+df['dairy']+'<br>'+\
    'Fruits '+df['total fruits']+' Veggies ' + df['total veggies']+'<br>'+\
    'Wheat '+df['wheat']+' Corn '+df['corn']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['total exports'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict( data=data, layout=layout )

url = py.plotly.plot( fig, filename='d3-cloropleth-map' )
