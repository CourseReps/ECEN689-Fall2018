#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 01:56:56 2018

@author: harinath
"""




import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
import seaborn as sns

winequality_combined_training_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-combined-training.csv')
winequality_combined_testing_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-combined-testing.csv')
winequality_combined_prediction_df = pd.read_csv('/Users/harinath/Documents/GitHub/ECEN689-Fall2018/Challenges/4Files/winequality-combined-sample.csv')

X = winequality_combined_training_df.iloc[:,1:12]
Y = winequality_combined_training_df.iloc[:,12:]
X_test = winequality_combined_testing_df.iloc[:,1:12]

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(X, Y)
pred1 = clf1.predict(X_test)
pred_df1 = pd.DataFrame(pred1)
pred_df1.to_csv('/Users/harinath/Desktop/DecisionTreeClassifier.csv', header=None)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf1, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("WineClassifier")


feature_importances = pd.DataFrame(clf.feature_importances_,
                               index = X.columns,
                                columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)

A = feature_importances['importance'].tolist()

#A = feature_importances[feature_importances.columns.difference(['importance'])]
#data = [go.Bar(x=feature_importances.index,
#            y=feature_importances.loc[:, ['class']])]

Z=feature_importances.index.tolist()
x = [1, 2, 3, 4,5,6,7,8,9,10,11]
 
plt.plot(x,A,'ro')
plt.xticks(x, Z, rotation='vertical')
#plt.margins(0.2)
# Tweak spacing to prevent clipping of tick-labels
#plt.subplots_adjust(bottom=0.5)
#ax = plt.subplots()
#ax.set_xticklabels(rotation = (45), fontsize = 10, va='bottom', ha='left')
plt.show()



var = 'type'

sns.boxplot(y=winequality_combined_training_df['fixed acidity'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['volatile acidity'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['citric acid'],x=winequality_combined_training_df['type'])
plt.show();
plt.show();
sns.boxplot(y=winequality_combined_training_df['residual sugar'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['chlorides'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['free sulfur dioxide'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['total sulfur dioxide'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['density'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['pH'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['sulphates'],x=winequality_combined_training_df['type'])
plt.show();
sns.boxplot(y=winequality_combined_training_df['alcohol'],x=winequality_combined_training_df['type'])
plt.show();