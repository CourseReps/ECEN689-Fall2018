import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def cv_optimize(clf, parameters, X, Y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(df2_X.values,labels)
    print("BEST", gs.best_params_, gs.best_score_, gs.grid_scores_)
    best = gs.best_estimator_
    score = gs.best_score_
    return best


def do_classify(clf, parameters,X, Y, X_test, score_func=None, n_folds=5, n_jobs=-1):
    if parameters:
        clf = cv_optimize(clf, parameters, df2_X.values, labels, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(df2_X.values, labels)
    training_accuracy = clf.score(df2_X.values, labels)
#     print "############# based on standard predict ################"
    print("Accuracy on training data: %0.2f" % (training_accuracy))
    prediction = clf.predict(df2_test.values)
    print(clf)
    print("########################################################")
    return prediction


df2_train = pd.read_csv("winequality-combined-training.csv")
df2_test = pd.read_csv("winequality-combined-testing.csv")
pd.set_option('display.max_columns', 1000)

df2_test = df2_test.iloc[:, 1:]

df2_train = df2_train.iloc[:, 1:]

df2_X = df2_train.iloc[:, 0:-1]
df2_Y = df2_train['type']
df2_Y = pd.DataFrame(df2_Y)
df2_Y.head()
df2_Y.type.unique()
labels = df2_Y.values
c, r = labels.shape
labels = labels.reshape(c,)
df2_Y.type.value_counts()

clfTree1 = tree.DecisionTreeClassifier()
clfTree2 = RandomForestClassifier()

parameters = {'max_depth': [15, 16, 17, 18, 19, 20, 21, 22, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40], 'min_samples_leaf': [1, 2, 3, 4, 5], 'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
#preds = do_classify(clfTree1, parameters,df2_X.values,labels, df2_test.values, n_folds = 3, score_func = 'accuracy')
preds = do_classify(clfTree2, parameters,df2_X.values,labels, df2_test.values, n_folds = 3, score_func = 'accuracy')

predictions = []

for i in range(len(preds)):
    predictions.append([i, preds[i]])

pred = pd.DataFrame(predictions,columns=['Id', 'type'])
pred.to_csv('winequality-combined-solution.csv', index=False)
