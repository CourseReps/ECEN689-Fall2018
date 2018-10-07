# Installed packages:
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLarsCV, LassoCV, RidgeCV,\
    LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Standard library:
import os.path
import time

# Define paths. TODO: Update when migrating into challenge folder.
# Input paths first:
IN_DIR = os.path.join('..', '..', 'Challenges', '4Files')
w = 'winequality'
# Combined:
COMBINED_TRAIN = os.path.join(IN_DIR, w + '-combined-training.csv')
COMBINED_TEST = os.path.join(IN_DIR, w + '-combined-testing.csv')
# Red:
RED_TRAIN = os.path.join(IN_DIR, w + '-red-training.csv')
RED_TEST = os.path.join(IN_DIR, w + '-red-testing.csv')
# White:
WHITE_TRAIN = os.path.join(IN_DIR, w + '-white-training.csv')
WHITE_TEST = os.path.join(IN_DIR, w + '-white-testing.csv')

# Now output paths:
OUT_DIR = '.'
PART1_PREDICTIONS = os.path.join(OUT_DIR, w + '-white-solution.csv')
PART1_COEFFICIENTS = os.path.join(OUT_DIR, w + '-white-parameters.csv')

# Use consistent seed.
SEED = 1234


def part_1():
    """Linear regression on the white wine data."""
    print('Running part 1...', end='\n\n')

    # Load training and testing data.
    train_df = pd.read_csv(WHITE_TRAIN, index_col='Id')
    test_df = pd.read_csv(WHITE_TEST, index_col='Id')

    # Grab X and y for easy access.
    train_size = 0.75
    test_size = 0.25
    train, test = train_test_split(train_df, train_size=train_size,
                                   test_size=test_size, random_state=SEED)
    X_train = train.drop(['quality'], axis=1)
    y_train = train['quality']
    X_test = test.drop(['quality'], axis=1)
    y_test = test['quality']

    print('Data loaded and split {}/{} train/test.'.format(train_size,
                                                           test_size),
          end='\n\n')

    # Initialize output dictionary.
    out = {}

    # Start with simple linear regression.
    print('Running LinearRegression...', end='')
    out['LinearRegression'] = {}
    t0 = time.time()
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train, y_train)
    p_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, p_lr)
    t1 = time.time()
    out['LinearRegression']['mse'] = mse_lr
    out['LinearRegression']['model'] = lr
    out['LinearRegression']['prediction'] = p_lr
    print('Done in {:.2f} seconds.'.format(t1 - t0), end='\n')

    # Try LassoCV.
    print('Running LassoCV...', end='')
    out['LassoCV'] = {}
    # Using 5-fold as that will be the new default soon.
    t0 = time.time()
    lcv = LassoCV(eps=1e-5, n_alphas=1000, cv=5, n_jobs=-1, random_state=SEED)
    lcv.fit(X_train, y_train)
    p_lcv = lcv.predict(X_test)
    mse_lcv = mean_squared_error(y_test, p_lcv)
    t1 = time.time()
    out['LassoCV']['mse'] = mse_lcv
    out['LassoCV']['model'] = lcv
    out['LassoCV']['prediction'] = p_lcv
    print('Done in {:.2f} seconds.'.format(t1 - t0), end='\n')

    # Try LassoLarsCV
    print('Running LassoLarsCV...', end='')
    out['LassoLarsCV'] = {}
    t0 = time.time()
    llcv = LassoLarsCV(cv=5, normalize=False, n_jobs=-1)
    llcv.fit(X_train, y_train)
    p_llcv = llcv.predict(X_test)
    mse_llcv = mean_squared_error(y_test, p_llcv)
    t1 = time.time()
    out['LassoLarsCV']['mse'] = mse_llcv
    out['LassoLarsCV']['model'] = llcv
    out['LassoLarsCV']['prediction'] = p_llcv
    print('Done in {:.2f} seconds.'.format(t1 - t0), end='\n')

    # Try RidgeCV
    print('Running RidgeCV...', end='')
    out['RidgeCV'] = {}
    # TODO: Should refine our alpha selection based on which one the
    # cross validation selects.
    t0 = time.time()
    rcv = RidgeCV(alphas=(0.01, 0.1, 10, 100, 1000), cv=5)
    rcv.fit(X_train, y_train)
    p_rcv = rcv.predict(X_test)
    mse_rcv = mean_squared_error(y_test, p_rcv)
    t1 = time.time()
    out['RidgeCV']['mse'] = mse_rcv
    out['RidgeCV']['model'] = rcv
    out['RidgeCV']['prediction'] = p_rcv
    print('Done in {:.2f} seconds.'.format(t1 - t0), end='\n\n')

    print('Evaluating models...')
    # Initialize for plotting.
    x = np.arange(1, lr.coef_.shape[0]+1)

    # Track best MSE. Initialize to infinity so we're always below it.
    best_mse = np.inf
    best_model = None

    # Loop to print results, plot results, and write the best prediction
    # to file.
    for key, value in out.items():
        # Printing:
        print(key + ' MSE: {:.2f}, '.format(value['mse']), end='')
        try:
            print('Alpha: {:.3f}'.format(value['model'].alpha_), end='')
        except AttributeError:
            pass

        try:
            alphas = value['model'].alphas_
            mmm = (np.min(alphas), np.median(alphas), np.max(alphas))
            print(', Alphas (min, med, max): ({:.2E}, {:.2E}, {:.2E})'.format(
                mmm[0], mmm[1], mmm[2]), end='')
        except AttributeError:
            pass

        print('')

        # Track best:
        if value['mse'] < best_mse:
            best_model = key
            best_mse = value['mse']

        # Plotting:
        plt.plot(x, value['model'].coef_)

    # Add legend, show plot.
    plt.legend(out.keys())
    plt.show()

    # Notify best model.
    print('\nBest model is {}.'.format(best_model))

    # Retrain with all the data.
    print('Retraining {} with all the data...'.format(best_model), end='')
    t0 = time.time()
    out[best_model]['model'].fit(X=train_df.drop(['quality'], axis=1),
                                 y=train_df['quality'])
    t1 = time.time()
    print('Done in {:.2f} seconds.'.format(t1-t0))

    # Write the best to file.
    best_predictions = pd.Series(out[best_model]['model'].predict(test_df),
                                 index=test_df.index, name='quality')
    best_predictions.to_csv(PART1_PREDICTIONS, header=True)
    best_coefficients = pd.Series(out[best_model]['model'].coef_,
                                  index=X_train.columns)
    pd.DataFrame(best_coefficients).T.to_csv(PART1_COEFFICIENTS, index=False)
    print('\nPrediction and coefficients written to file.')


if __name__ == '__main__':
    part_1()
