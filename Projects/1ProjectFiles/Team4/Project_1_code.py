# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Contributor:- Sambandh, Drupad, Kishan, Harish
"""

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import io
import random
import re

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


dtype_dic_zip= {'zipcode': str}
income = pd.read_csv('D:/Study/3rd Sem/ECEN 689/Project 1/16zpallnoagi.csv', engine='python',dtype = dtype_dic_zip)

income = income.rename(index=str,columns={'ZIPCODE':'zip'})
income

dtype_dic_combined = {'zip': str,'county':str}
zipp = pd.read_csv('D:/Study/3rd Sem/ECEN 689/Project 1/zipfips.csv',dtype = dtype_dic_combined)

zipp = zipp.rename(index=str,columns={'county':'FIPS'})

zipp.head()

#e1 = pd.merge(zipp, income, on='zip', how='outer')
zipp['zip']= zipp['zip'].astype(int) 
e1= pd.merge(income, zipp, on='zip', how='inner')
e1

e2=e1.drop(['STATEFIPS','STATE'], axis=1)

e4 = e2[['N2','A02650','A00100','FIPS','zip']] 

e5 = e4.rename(index=int,columns={'N2':'pop','A02650':'total_income','A00100':'adj_gross_income'})

dtype_dic_FIPS= {'FIPS': str}
health = pd.read_csv('D:/Study/3rd Sem/ECEN 689/Project 1/food1.csv',dtype = dtype_dic_FIPS)

f={'total_income':'mean','adj_gross_income':'mean','FIPS':'first', 'pop':'first'}
e6=e5.groupby('zip',as_index=False).agg(f)

wm= lambda x: np.average(x,weights=e6.loc[x.index,'pop'])
f={'total_income':{'weighted_mean':wm },'adj_gross_income':{'weighted_mean':wm },'pop':'sum'}
e8=e6.groupby('FIPS',as_index=False).agg(f)
e8

e9= pd.merge(e8, health, on='FIPS', how='inner')
e9['FIPS']= e9['FIPS'].astype(int) 

e10 = e9.drop(e9.columns[0],axis=1)
e10.columns=['FIPS', 'total_income','adj_gross_income','Population','State','County','PCT_DIABETES_ADULTS08','PCT_DIABETES_ADULTS13','PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13','PCT_HSPA15','RECFAC09','RECFAC14','PCH_RECFAC_09_14','RECFACPTH09','RECFACPTH14','PCH_RECFACPTH_09_14']

app_train = e10

app_train = app_train.drop(app_train.columns[0], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(app_train, test_size=0.01)
#X_test.reset_index(drop=True)
#X_train.reset_index(drop=True)
X_train["State"] = X_train["State"].astype('category')
X_train["County"] = X_train["County"].astype('category')
X_test["State"] = X_test["State"].astype('category')
X_test["County"] = X_test["County"].astype('category')

result_train= X_train.drop(['FIPS','total_income','Population','State','County','PCT_DIABETES_ADULTS08','PCT_OBESE_ADULTS08','RECFAC09','PCT_HSPA15','PCH_RECFAC_09_14','RECFACPTH09','RECFACPTH14','PCH_RECFACPTH_09_14'], axis = 1)

Y_train = result_train["adj_gross_income"]
Y_train = pd.DataFrame(Y_train,columns=['adj_gross_income'])
result_train = result_train.drop(["adj_gross_income"], axis=1)

lm = LinearRegression()
model = lm.fit(result_train,Y_train)

print(model.intercept_)  
print(model.coef_)


plt.matshow(result_train.corr())
pd.scatter_matrix(result_train, figsize=(4, 4))
plt.show()
