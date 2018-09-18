import pandas as pd
import numpy as np
import os

print(os.getcwd())
os.chdir("../../../../Challenges/1Files/")
print(os.getcwd())

student_identity = 'littlebass09'
filenameToRead = '1challenge1activity_' + student_identity + '.csv'
filename = '1challenge1activity_' + student_identity + '.csv'
sample_df = pd.read_csv(filenameToRead)

list = []
for row in range(sample_df.shape[0]):
    array =  [sample_df['Sample 0'][row],sample_df['Sample 1'][row],
        sample_df['Sample 2'][row],sample_df['Sample 3'][row],
        sample_df['Sample 4'][row],sample_df['Sample 5'][row],
        sample_df['Sample 6'][row],sample_df['Sample 7'][row],
        sample_df['Sample 8'][row],sample_df['Sample 9'][row],
        sample_df['Sample 10'][row],sample_df['Sample 11'][row]]
    list.append([np.mean(array),np.var(array)])

for row in range(sample_df.shape[0]):
    sample_df.iloc[row, sample_df.columns.get_loc('Mean')] = list[row][0]
    sample_df.iloc[row, sample_df.columns.get_loc('Variance')] = list[row][1]

sample_df = sample_df.drop('Unnamed: 0', 'columns')
sample_df.to_csv(filename, sep=',', encoding='utf-8')
print('finished')
