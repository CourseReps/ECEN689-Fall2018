import pandas as pd
import os
from inspect import getsourcefile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier as BDT
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

# locations
homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}4Files{0}'.format(os.path.sep)
targetDir = homeDir

# tree limits to test
tree_limits = [i for i in range(1, 30)]

# number of bins
cv_bins = 100

if __name__=='__main__':

    # load data
    wine_training = shuffle(pd.read_csv(sourceDir + 'winequality-combined-training.csv', index_col='Id'))
    # print(wine_training.shape)
    wine_testing = pd.read_csv(sourceDir + 'winequality-combined-testing.csv', index_col='Id')
    # print(wine_testing.shape)

    # split to train and verify sets for cross-validation
    bin_size = wine_training.shape[0]//cv_bins
    assert bin_size > 0, 'too many bins for too few training rows'
    me_train_linear = 0
    me_test_linear = 0
    me_train = [0 for _ in tree_limits]
    me_test = [0 for _ in tree_limits]
    me_train_lda = [0 for _ in tree_limits]
    me_test_lda = [0 for _ in tree_limits]
    for bin in range(cv_bins):
        # set up
        test_rows = [r for r in range(bin*bin_size, (bin+1)*bin_size)]
        light_train = wine_training.drop(test_rows)
        light_test = wine_training.iloc[test_rows]

        # transform set for best binary decision tree performance
        X_train = light_train.drop('type', axis=1)
        X_test = light_test.drop('type', axis=1)

        lda = LDA()
        lda.fit(X_train, light_train['type'])
        me_train_linear += (1-lda.score(X_train, light_train['type']))/cv_bins
        me_test_linear += (1-lda.score(X_test, light_test['type']))/cv_bins

        # train binary decision trees of various sizes
        for i, tl in enumerate(tree_limits):
            bdt = BDT(max_depth=tl, presort=True)
            bdt.fit(X_train, light_train['type'])
            me_train[i] += (1-bdt.score(X_train, light_train['type']))/cv_bins
            me_test[i] += (1-bdt.score(X_test, light_test['type']))/cv_bins

            lda = LDA()
            X_train = lda.fit_transform(X_train, light_train['type'])
            X_test = lda.transform(X_test)
            bdt = BDT(max_depth=tl, presort=True)
            bdt.fit(X_train, light_train['type'])
            me_train_lda[i] += (1-bdt.score(X_train, light_train['type']))/cv_bins
            me_test_lda[i] += (1-bdt.score(X_test, light_test['type']))/cv_bins
    # LDA performance alone
    print('LDA model error Train:', me_train_linear, 'Test:', me_test_linear)

    # check best of each tree
    print('Best raw Train:', min(me_train), 'Test:', min(me_test))
    print('Best lda Train:', min(me_train_lda), 'Test:', min(me_test_lda))

    # compare methods
    plt.figure()
    plt.hlines(me_train_linear, tree_limits[0], tree_limits[-1], label='linear model train')
    plt.hlines(me_test_linear, tree_limits[0], tree_limits[-1], label='linear model test')
    plt.semilogy(tree_limits, me_train, label='raw train score')
    plt.semilogy(tree_limits, me_test, label='raw test score')
    plt.semilogy(tree_limits, me_train_lda, label='lda train score')
    plt.semilogy(tree_limits, me_test_lda, label='lda test score')
    plt.legend()
    plt.show()