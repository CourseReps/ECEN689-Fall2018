#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:18:50 2018

@author: khalednakhleh

Copying this work is prohibited under all circumstances.
"""

import pandas as pd

df = pd.read_csv("/Users/khalednakhleh/Documents/ecen_689/ECEN689-Fall2018" \
                 "/Challenges/1Files/1challenge1activity_khaled.jamal.csv", index_col=0)

print(df.head(2))

# Calculating Mean and Variance for the given .csv file

# axis=1 for calculating row-wise.
df["Mean"] = df.mean(axis=1) 

# calculating from the second column to the last column.
df["Variance"] = (df.iloc[:, 2:]).var(axis=1) 


df.to_csv('/Users/khalednakhleh/Documents/ecen_689/ECEN689-Fall2018' \
          '/Challenges/1Files/khalednakhleh/mean_and_var.csv')

# Final file can be found in challenges/1Files/khalednakhleh folder on GitHub,
# File name is mean_and_var.csv