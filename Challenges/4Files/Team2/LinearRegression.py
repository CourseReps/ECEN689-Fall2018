#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 01:56:56 2018

@author: harinath
"""




import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings 
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib
from itertools import chain, combinations
import statsmodels.api as sm


winequality_white_training_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-training.csv')
winequality_white_testing_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-testing.csv')
winequality_white_prediction_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-sample.csv')

X = winequality_white_training_df.iloc[:,1:12]
Y = winequality_white_training_df.iloc[:,12:]
X_test = winequality_white_testing_df.iloc[:,1:12]

regr = linear_model.LinearRegression()
regr.fit(X, Y)
Y_pred = regr.predict(X_test)


pred_df = pd.DataFrame(Y_pred)

pred_df.to_csv('/Users/harinath/Desktop/LinearRegression1.csv', header=None)

import statsmodels.api as sm
from scipy import stats


#to compute the pvalue of model to see all of them are critical or not

est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())

#correlation matrix to find if multicollinearity exists or not
k = 11 #number of variables for heatmap
corrmat = X.corr()
cols = corrmat.nlargest(k, 'fixed acidity')['fixed acidity'].index
cm = np.corrcoef(X[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

cv = KFold(n_splits=10)
from sklearn.feature_selection import RFE

i=0
test_f1Score = []
test_accuracy = []
for train, test in cv.split(X, Y):
    X_train, X_test = X.values[train], X.values[test]
    Y_train, Y_test = Y.values[train], Y.values[test]
    print(X_train.dtype.names)
    def error_comp(y_actual, y_pred):
        error=0
        for i in range(len(y_actual)) :
            error=error+pow((y_actual[i]-y_pred[i]),2)
        return error
    mac_=10000
    max_=0
    train_errors = list()
    test_errors = list()
    for item in range(1,12):
        fecv = RFE(estimator=regr, step=1,
              n_features_to_select=item)
        fecv.fit(X_train, Y_train)
        test_predict=fecv.predict(X_test)
        train_predict=fecv.predict(X_train)
        train_errors.append(error_comp(Y_train,train_predict))
        test_errors.append(error_comp(Y_test, test_predict))
        if error_comp(Y_test, test_predict) < mac_:
            max_=item
            mac_= error_comp(Y_test, test_predict)
    print (max_,mac_)
    
    fecv = RFE(estimator=regr, step=1,
              n_features_to_select=max_)
    fecv.fit(X.values[train], Y_train)
    
    
    fecv.support_ 
    feature_importances = pd.DataFrame(fecv.support_ ,
                                   index = X.columns,
                                    columns=['importance'])

    print(feature_importances)
    fecv.ranking_

    
    X_train_final=fecv.transform(X.values[train])
    X_temp_final=fecv.transform(X_test)
    X2 = sm.add_constant(X_train_final)
    est = sm.OLS(Y_train, X2)
    est2 = est.fit()
    print(est2.summary())



#function to find best subset of features with best RSq


def best_subset(X, y):
    n_features = X.shape[1]
    subsets = chain.from_iterable(combinations(range(1,11), k+1) for k in range(1,11))
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

best_sub,adj_rsq = best_subset(X,Y)
sub2=list(best_sub)
lin_reg = sm.OLS(Y, X.iloc[:, sub2]).fit()
score = lin_reg.rsquared_adj
print(lin_reg.summary())
print(score)

est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())

X_test = winequality_white_testing_df.iloc[:,1:12]

regr = linear_model.LinearRegression()
regr.fit(X.iloc[:, sub2], Y)
Y_pred = regr.predict(X_test.iloc[:, sub2])

pred_df = pd.DataFrame(Y_pred)

pred_df.to_csv('/Users/harinath/Desktop/LinearRegression.csv', header=None)

winequality_white_training_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-training.csv')
winequality_white_testing_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-testing.csv')
winequality_white_prediction_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-red-sample.csv')

X = winequality_white_training_df.iloc[:,1:12]
Y = winequality_white_training_df.iloc[:,12:]
X_test = winequality_white_testing_df.iloc[:,1:12]