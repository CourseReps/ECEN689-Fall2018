# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

student_identity = 'harishk1908'
filenameToRead = '1challenge1activity_' + student_identity + '.csv'
filenameToWrite = 'o1challenge1activity_' + student_identity + '.csv'
filename = '1challenge1activity_' + student_identity + '.csv'
sample_df = pd.read_csv(filenameToRead)

means = []
variances = []
meanIndex = 1
varianceIndex = 2
for rowIndex in range(len(sample_df)):
    sample_values = []
    for columnIndex in range(12):
        sample_values.append(sample_df.loc[rowIndex]['Sample ' + str(columnIndex)])
    mean = np.mean(sample_values)
    variance = np.var(sample_values)
    sample_df.iloc[rowIndex, meanIndex] = mean
    sample_df.iloc[rowIndex, varianceIndex] = variance

sample_df = sample_df.drop('Unnamed: 0', axis=1)
sample_df.to_csv(filenameToWrite)