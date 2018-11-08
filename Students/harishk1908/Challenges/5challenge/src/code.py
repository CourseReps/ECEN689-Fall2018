# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

data_path = '../data/'
data_extension = '.csv'
train_file = '5challenge_training_harishk1908'
test_file = '5challenge_testing_harishk1908'

def fileReader(file_name):    
    return pd.read_csv(data_path + file_name + data_extension, delimiter=',').values[:,1:]

def getDataAndTargets(matrix):
    return matrix[:, 1:], matrix[:, 0]

def getDataFromFile(file_name):
    matrix = fileReader(file_name)
    return getDataAndTargets(matrix)

def makePointGrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contour(xx, yy, Z, **params)
    return out


trainData, trainTargets = getDataFromFile(train_file)
testData, _ = getDataFromFile(test_file)

param_gridLinear = [{'C': 10**np.linspace(-3, 3, 7), 'kernel': ['linear']}]
param_gridRBF =[{'C': 10**np.linspace(-4, 4, 32), 'gamma': 10**np.linspace(-4, -1, 32), 'kernel': ['rbf']}]
param_gridSigmoid = [{'C': 10**np.linspace(-3, 3, 16), 'gamma': 10**np.linspace(-4, -1, 16), 'kernel': ['sigmoid']}]
param_gridPoly = [{'C': 10**np.linspace(-3, 3, 16), 'gamma': 10**np.linspace(-4, -1, 16), 'kernel': ['poly'], 'degree': [2,3,4,5,6]}]


svc = svm.SVC()

clfLinear = GridSearchCV(svc, param_gridLinear, cv=5, verbose=1, n_jobs = 1, refit=True)
clfRBF = GridSearchCV(svc, param_gridRBF, cv=5, verbose=1, n_jobs = 1, refit=True)
clfSigmoid = GridSearchCV(svc, param_gridSigmoid, cv=5, verbose=1, n_jobs = 1, refit=True)
clfPoly = GridSearchCV(svc, param_gridPoly, cv=5, verbose=1, n_jobs = 1, refit=True)

def fitAndPrintPerformance(cvFitter):
    cvFitter.fit(trainData, trainTargets)
    print(cvFitter.best_estimator_)
    print('Score: ', cvFitter.best_score_)

print('Linear')
fitAndPrintPerformance(clfLinear)
print('RBF')
fitAndPrintPerformance(clfRBF)
print('Sigmoid')
fitAndPrintPerformance(clfSigmoid)
print('Poly')
fitAndPrintPerformance(clfPoly)

plt.scatter(trainData[:,0], trainData[:,1], c=trainTargets)

X0, X1 = trainData[:, 0], trainData[:, 1]
xx, yy = makePointGrid(X0, X1)

clf = clfRBF #Hardcode, since I've already determined which one performs best.
scores = cross_val_score(clf.best_estimator_, trainData, trainTargets, cv=10)
plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
plt.xlabel('x1')
plt.ylabel('x2')

plt.savefig('../plots/decision_boundary.eps', type='eps')
plt.show()
print(clf.best_estimator_)
print('Cross validation scores: ', scores) 

#clf= clf.best_estimator_
testResults = clf.predict(testData)

testDf = pd.read_csv(data_path + test_file + data_extension, delimiter=',')
testDf['Class'] = testResults.astype('int')
testDf = testDf.drop('Unnamed: 0', axis=1)
testDf.to_csv(data_path + test_file + 'out' + data_extension, index=True)