#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 23:05:15 2018

@author: harinath
"""

import pandas as pd
#from google.colab import files 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings 
warnings.filterwarnings('ignore')

from itertools import chain, combinations
import statsmodels.api as sm

trainData = '/Users/harinath/Desktop/ECEN689/Health.csv'
health_df = pd.read_csv(trainData)

#hud_df = pd.read_csv(io.StringIO(uploaded_hud['COUNTY_ZIP_122016.csv'].decode('cp1252')))
health_df.head(10)

trainDataSource = '/Users/harinath/Downloads/FIPSFilteredData.csv'
income_df = pd.read_csv(trainDataSource)
income_df.head(10)

combined= pd.merge(health_df, income_df, on='FIPS')
print(combined.shape)

trainData = '/Users/harinath/Desktop/ECEN689/Local.csv'
local_df = pd.read_csv(trainData)

trainData = '/Users/harinath/Desktop/ECEN689/prices_taxes1.csv'
pt_df = pd.read_csv(trainData)

temp= pd.merge(combined, local_df, on='FIPS')
print(temp.shape)

combined_locpt= pd.merge(temp, pt_df, on='FIPS')
print(combined_locpt.shape)

sns.distplot(combined['A02650'])
plt.show();

sns.boxplot(combined['A02650'])
plt.show();

#Z = combined.loc[:,['A02650','PCT_DIABETES_ADULTS08','PCT_DIABETES_ADULTS13',
#                    'PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13','PCH_RECFAC_09_14',
#'PCH_RECFACPTH_09_14']]



Z = combined.loc[:,['A02650','PCT_DIABETES_ADULTS08','PCT_DIABETES_ADULTS13',
                    'PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13','PCT_HSPA15	',
                    'RECFAC09','RECFAC14',	
                    'RECFACPTH09','RECFACPTH14']]
k = 12 #number of variables for heatmap
corrmat = Z.corr()
cols = corrmat.nlargest(k, 'A02650')['A02650'].index
cm = np.corrcoef(Z[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

X = combined["RECFAC09"] ## X usually means our input variables (or independent variables)
y = combined["A02650"]
print(linregress(X, y))
#state_df=combined.groupby(['State'],sort=False).mean()
#state_df=state_df.drop(['DC'])
#print(state_df.shape)

from sklearn.cluster import KMeans
# Incorrect number of clusters
X = combined[['PCT_DIABETES_ADULTS08','A02650','PCT_OBESE_ADULTS08']] ## X usually means our input variables (or independent variables)
y = combined["A02650"] ## Y usually means our output/dependent variable

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.scatter(X['PCT_DIABETES_ADULTS08'], X['A02650'])
plt.title("Original")

state_df=combined.groupby(['State'],sort=False).mean()
state_df=state_df.drop(['DC'])
#print(state_df.shape)
y_pred = KMeans(n_clusters=6, random_state=5).fit_predict(X)

plt.figure(figsize=(12, 12))

#plt.subplot(222)
#plt.scatter(state_df['PCT_DIABETES_ADULTS08'], state_df['A02650'], state_df['PCT_OBESE_ADULTS08'], c=y_pred)
#plt.title("Number of Blobs")

#filter based on criteria
# <100000 1654 observations : no relationship between them at all , >500000 298

#criteria_1 = combined['A02650'] <= 100000
#combined=combined[criteria_1]


X = combined["PCT_DIABETES_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined["A02650"] ## Y usually means our output/dependent variable
#X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print(model.summary())


print(linregress(X, y))

criteria_1 = combined_locpt['A02650'] >= 100000
criteria_2 = combined_locpt['A02650'] < 100000
combined_high=combined_locpt[criteria_1]
combined_low=combined_locpt[criteria_2]

X = combined_high["PCT_DIABETES_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined_high["A02650"]
print(linregress(X, y))

X = combined_high["PCT_OBESE_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined_high["A02650"]
print(linregress(X, y))

X = combined_high["PCT_OBESE_ADULTS13"] ## X usually means our input variables (or independent variables)
y = combined_high["A02650"]
print(linregress(X, y))

X = combined_low["PCT_DIABETES_ADULTS13"] ## X usually means our input variables (or independent variables)
y = combined_low["A02650"]
print(linregress(X, y))

X = combined_low["PCT_OBESE_ADULTS08"] ## X usually means our input variables (or independent variables)
y = combined_low["A02650"]
print(linregress(X, y))

X = combined_low["PCT_OBESE_ADULTS13"] ## X usually means our input variables (or independent variables)
y = combined_low["A02650"]
print(linregress(X, y))
combined_locpt.to_csv('Health_IRS.csv')


 
Z = combined_locpt.loc[:,['A02650','PCT_DIABETES_ADULTS08','PCT_DIABETES_ADULTS13',
                    'PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13','PCT_HSPA15	',
                    'RECFAC09','RECFAC14',	
                    'RECFACPTH09','RECFACPTH14',
                    'FMRKTPTH09','FMRKTPTH16','DIRSALES_FARMS07',	
                    'DIRSALES_FARMS12','CSA07',
                    'CSA12','VEG_FARMS07','VEG_FARMS12',
                    'MILK_PRICE10','SODA_PRICE10','MILK_SODA_PRICE10',
                    'SODATAX_STORES14','SODATAX_VENDM14',
                    'CHIPSTAX_STORES14','CHIPSTAX_VENDM14','FOOD_TAX14'
                    ]]
k = 30 #number of variables for heatmap
corrmat = Z.corr()
cols = corrmat.nlargest(k, 'PCT_DIABETES_ADULTS08')['PCT_DIABETES_ADULTS08'].index
cm = np.corrcoef(Z[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

Z1 = combined_locpt.loc[:,['A02650','PCT_DIABETES_ADULTS08',
                   'PCT_OBESE_ADULTS08','PCT_HSPA15	',
                    'RECFAC09',	
                    'RECFACPTH09',
                    'FMRKTPTH09',	
                    'DIRSALES_FARMS12',
                    'CSA12','VEG_FARMS12',
                    'MILK_SODA_PRICE10','SODA_PRICE10'
                    ]]
k = 30 #number of variables for heatmap
corrmat = Z1.corr()
cols = corrmat.nlargest(k, 'PCT_DIABETES_ADULTS08')['PCT_DIABETES_ADULTS08'].index
cm = np.corrcoef(Z1[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


criteria_1 = Z['RECFAC09'] > 0
Z=Z[criteria_1]


plt.figure(figsize=(12, 12))
plt.scatter(Z['RECFAC09'], Z['A02650'])
plt.title("Original")

#function to find best subset of features with best RSq


def best_subset(X, y):
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(range(1,20), k+1) for k in range(1,20))
    best_score = -np.inf
    best_subset = None
    for subset in subsets:
        sub= list(subset)
        lin_reg = sm.OLS(y, X.iloc[:, sub]).fit()
        score = lin_reg.rsquared_adj
        if score > best_score:
            best_score, best_subset = score, subset
    return best_subset, best_score


#using best subset to build a model and get its parameters 
Y= Z['PCT_DIABETES_ADULTS08']
X= Z[['A02650','RECFAC09','RECFAC14',	
                    'RECFACPTH14',
                    'FMRKTPTH09','FMRKTPTH16','DIRSALES_FARMS07',	
                    'DIRSALES_FARMS12','CSA07',
                    'CSA12','VEG_FARMS07','VEG_FARMS12',
                    'MILK_PRICE10','SODA_PRICE10','MILK_SODA_PRICE10',
                    'SODATAX_STORES14','SODATAX_VENDM14',
                    'CHIPSTAX_STORES14','CHIPSTAX_VENDM14','FOOD_TAX14'
                    ]]
#best_sub,adj_rsq = best_subset(X,Y)
#sub2=list(best_sub)
#lin_reg = sm.OLS(Y, X.iloc[:, sub2]).fit()
#score = lin_reg.rsquared_adj
#print(lin_reg.summary())
#print(score)

#est = sm.OLS(Y, X)
#est2 = est.fit()
#print(est2.summary())

#X_test = winequality_white_testing_df.iloc[:,1:12]

#regr = linear_model.LinearRegression()
#regr.fit(X.iloc[:, sub2], Y)
#Y_pred = regr.predict(X_test.iloc[:, sub2])

#pred_df = pd.DataFrame(Y_pred)

#socio economic data too

Z1 = combined_locpt.loc[:,['FIPS','A02650','PCT_DIABETES_ADULTS08','PCT_DIABETES_ADULTS13',
                    'PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13','PCT_HSPA15	',
                    'RECFAC09','RECFAC14',	
                    'RECFACPTH09','RECFACPTH14',
                    'FMRKTPTH09','FMRKTPTH16','DIRSALES_FARMS07',	
                    'DIRSALES_FARMS12','CSA07',
                    'CSA12','VEG_FARMS07','VEG_FARMS12',
                    'MILK_PRICE10','SODA_PRICE10','MILK_SODA_PRICE10',
                    'SODATAX_STORES14','SODATAX_VENDM14',
                    'CHIPSTAX_STORES14','CHIPSTAX_VENDM14','FOOD_TAX14'
                    ]]
trainData = '/Users/harinath/Desktop/ECEN689/socioecon.csv'
socio_df = pd.read_csv(trainData)

temp= pd.merge(Z1, socio_df, on='FIPS')
print(temp.shape)

temp.to_csv('Health_Combined.csv')

temp=temp[temp.columns.difference(['State',
                        'County'])]

k = 50 #number of variables for heatmap
corrmat = temp.corr()
cols = corrmat.nlargest(k, 'PCT_DIABETES_ADULTS08')['PCT_DIABETES_ADULTS08'].index
cm = np.corrcoef(temp[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

print("Correlation Matrix")
#print(temp.corr())
#print()

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df,var, n=5):
    au_corr = df.corr().abs().unstack()
    #print(au_corr)
    #labels_to_drop = get_redundant_pairs(df)
    #au_corr = au_corr['A02650'].drop(labels=labels_to_drop).sort_values(ascending=False)
    au_corr = au_corr[var].sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
var1='A02650'
print(get_top_abs_correlations(temp,var1, 10))

trainData = '/Users/harinath/Desktop/ECEN689/Merged_Data.csv'
overall_df = pd.read_csv(trainData)

overall_df=overall_df[overall_df.columns.difference(['State',
                        'County'])]

Y= overall_df['PCT_DIABETES_ADULTS08']
X= overall_df[overall_df.columns.difference(['FIPS','PCT_DIABETES_ADULTS08',
                                             'PCT_DIABETES_ADULTS13',
                                             'PCT_OBESE_ADULTS08','PCT_OBESE_ADULTS13'])]

#best_sub,adj_rsq = best_subset(X,Y)
#sub2=list(best_sub)
#lin_reg = sm.OLS(Y, X.iloc[:, sub2]).fit()
#score = lin_reg.rsquared_adj
#print(lin_reg.summary())
#print(score)

#est = sm.OLS(Y, X)
#est2 = est.fit()
#print(est2.summary())
