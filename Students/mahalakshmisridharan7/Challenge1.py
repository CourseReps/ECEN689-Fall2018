import pandas as pd
import numpy as np

student_identity = 'mahalakshmi'

#File to read
filename = '1challenge1activity_' + student_identity + '.csv'

#Read CSV file into dataframe
df = pd.read_csv('../../Challenges/1Files/' + filename)

#Create a new dataframe with only the Sample Data columns
sample_df=df.iloc[:,3:15]

#Converting dataframe to array
array_df=sample_df.values

#Calculating mean and variance
mean=np.mean(array_df,axis=1,dtype=np.float64)
variance=np.var(array_df,axis=1,dtype=np.float64)

#Update mean and variance in dataframe
df["Mean"]=mean
df["Variance"]=variance

#Write to csv file excluding index
sample_df.to_csv('../../Challenges/1Files/'+filename, sep=',', encoding='utf-8',index=False)
