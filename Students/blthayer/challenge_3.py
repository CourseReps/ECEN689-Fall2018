########################################################################
# IMPORTS
# Standard library
from os.path import join
# Installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

########################################################################
# CONSTANTS
IN_DIR = join('..', '..', 'Challenges', '3Files')
OUT_DIR = '.'

TEST_FILE = join(IN_DIR, 'population_testing.csv')
TRAIN_FILE = join(IN_DIR, 'population_training.csv')

SEED = 42

if __name__ == '__main__':
    ####################################################################
    # Read files
    train_df = pd.read_csv(TRAIN_FILE, encoding='cp1252').dropna(axis=0)
    test_df = pd.read_csv(TEST_FILE, encoding='cp1252').dropna(axis=0)

    # Get matrices.
    train_mat = train_df.drop(['Country Name'], axis=1).values.T.astype(int)
    test_mat = train_df.drop(['Country Name'], axis=1).values.T.astype(int)

    # For now, focus on one country (column 1). Later, we'll need to
    # generalize and operate on all countries.

    # Initialize alphas
    alphas = 10 ** np.linspace(10, -2, 5) * 0.5
    # Track coefficients
    coeff = []

    # Loop over alphas and fit.
    for a in alphas:
        # Initialize a Lasso object
        lasso = Lasso(alpha=a)
        # Fit.
        lasso.fit(train_mat[:, 1:], train_mat[:, 0])
        # Extract these coefficients.
        c = lasso.coef_
        # Track coefficients if we have <= 5 "non-zero" elements.
        non_zero = ~np.isclose(c, 0)
        if np.count_nonzero(non_zero) <= 5:
            # Ensure values that were close to 0 are exactly 0.
            c[~non_zero] = 0
            # Track.
            coeff.append(c)
        else:
            # We had too many non-zero coefficients.
            coeff.append(None)


    pass
