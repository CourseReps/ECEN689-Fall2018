#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:50:28 2018

Name: khalednakhleh

income file taken from:
    https://www.irs.gov/statistics/soi-tax-stats-county-data-2016
    Removed Kalawo county from the health dataset since it didn't have
    an income data values.
"""

import pandas as pd
import numpy as np

health = pd.read_csv("health_data.csv")
income = pd.read_csv("income_2013.csv")


print("Initial dimensions for health and income datasets: \n")
print(health.shape)
print(income.shape)

###################################################
income_adjust = income[~((income["COUNTYFIPS"] == 0))]
new_header_1 = income_adjust.iloc[0, :]
income_adjust = income_adjust[1:]

income_adjust.columns = new_header_1
print(income_adjust.columns)
health_adjust = health[~((health["County"] == "Kalawao"))]

new_header_2 = health_adjust.iloc[0, :]
health_adjust = health_adjust[1:]

health_adjust.columns = new_header_2

new_index = np.arange(health_adjust.shape[0])
income_adjust.index = new_index
health_adjust.index = new_index

###################################################
print("\n--------------------------------")
print("after parsing dimensions for health and income datasets: \n")

print("health dimensions: " + str(health_adjust.shape))
print("income dimensions: " + str(income_adjust.shape))


income_adjust.to_csv("‎income_adjusted.csv")
health_adjust.to_csv("‎health_adjusted.csv")


###################################################

# Combining the parsed health and income files into one file

combined = pd.concat([health_adjust, income_adjust], axis = 1)


combined.to_csv("combined_2013.csv", index = "Id")

print(combined.head())
print("\nParsed file dimensions: \n")
print(combined.shape)




 
 
 
 
 
