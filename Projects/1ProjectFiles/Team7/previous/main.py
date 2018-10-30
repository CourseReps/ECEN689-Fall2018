#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:40:47 2018

Name: khalednakhleh

Run the file parsing.py first to obtain the file combined.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    health_income = pd.read_csv("combined.csv")
    
    print(health_income.shape)
    print(health_income.head())
    
    
    # Grabbing the health data
    diabetes_2008 = health_income.iloc[:, 4]
    diabetes_2013 = health_income.iloc[:, 5]
    obesity_2008 = health_income.iloc[:, 6]
    obesity_2013 = health_income.iloc[:, 7] 
    joint_return = health_income.iloc[:, 22] 
    
    

    plt.scatter(joint_return, diabetes_2013)
    plt.show()
    
    plt.scatter(obesity_2008, obesity_2013)
    plt.show()

if __name__ == "__main__":
    main()

