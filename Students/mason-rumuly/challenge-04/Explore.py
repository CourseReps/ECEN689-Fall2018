import pandas as pd
import os
from inspect import getsourcefile
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso, LinearRegression

import matplotlib.pyplot as plt

from SGDRegressor import SGDRegressor

# best performance on latest settings, no regularization, 1000 epochs per learning rate:
# MSE Train: 0.628818335823 Test: 0.633627319819
# Score Train: 0.193460322035 Test: 0.187668296912

homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}4Files{0}'.format(os.path.sep)
targetDir = homeDir

cv_bins = 4
max_a_iter = 1000

if __name__=='__main__':

    # load data
    white_training = shuffle(pd.read_csv(sourceDir + 'winequality-white-training.csv', index_col='Id'))

    # split to train and verify sets for cross-validation
    bin_size = white_training.shape[0]//cv_bins
    assert bin_size > 0, 'too many bins for too few training rows'

    # conduct LASSO elimination of variables
    amin = 1e-10
    amax = None
    lasso = None
    X = white_training.drop('quality', axis=1)
    X /= X.mean()
    y = white_training['quality']
    y /= y.mean()
    excluded = X.columns.values
    train_scores = [0 for _ in range(len(X.columns.values))]
    test_scores = [0 for _ in range(len(X.columns.values))]
    for nv in range(len(X.columns.values)):
        for _ in range(max_a_iter):
            # update a
            a = amin*2 if amax is None else (amax + amin) / 2

            # train Lasso
            lasso = Lasso(a)
            lasso.fit(X, y)

            # check number of nonzero variables
            nz = np.count_nonzero(lasso.coef_)
            if nz == nv + 1:
                break
            elif nz <= nv:
                amax = a
            else:
                amin = a

        # reset min, alpha is counting down to allow additional
        amin = 1e-10

        # show the excluded variables
        new_ex = [X.columns.values[i] for i, c in enumerate(lasso.coef_) if c == 0]
        print(nv + 1, [e for e in excluded if not e in new_ex], [e for e in new_ex if not e in excluded])
        excluded = new_ex

        # construct cross-validation bins
        for bin in range(cv_bins):
            test_rows = [r for r in range(bin*bin_size, (bin+1)*bin_size)]
            light_train = white_training.drop(test_rows).drop(excluded, axis=1).drop('quality', axis=1)
            light_test = white_training.iloc[test_rows].drop(excluded, axis=1).drop('quality', axis=1)
            light_train_y = white_training['quality'].drop(test_rows)
            light_test_y = white_training['quality'].iloc[test_rows]

            # train regression
            lreg = LinearRegression()
            lreg.fit(light_train, light_train_y)

            # average scores
            train_scores[nv] += lreg.score(light_train, light_train_y)/cv_bins
            test_scores[nv] += lreg.score(light_test, light_test_y)/cv_bins
        
# plot
plt.figure()
nvars = [nv + 1 for nv in range(len(X.columns.values))]
plt.plot(nvars, train_scores, label='train')
plt.plot(nvars, test_scores, label='test')
plt.xlabel('Kept Variables')
plt.ylabel('CV Mean Score')
plt.legend()
plt.show()
