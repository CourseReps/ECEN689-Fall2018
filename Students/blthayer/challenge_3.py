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
# In/out paths.
IN_DIR = join('..', '..', 'Challenges', '3Files')
OUT_DIR = '.'

# Input files.
TEST_FILE = join(IN_DIR, 'population_testing.csv')
TRAIN_FILE = join(IN_DIR, 'population_training.csv')

# Output files.
PRED_OUT = join(OUT_DIR, '3population_predicted.csv')
COEFF_OUT = join(OUT_DIR, '3parameters.csv')

SEED = 42

COEFF_MAX = 5

def lasso_for_alphas(alphas, X, y, X_test):
    # Track best score for this country.
    best_score = np.inf

    # Initialize best_coefficients
    best_coeff = np.zeros(X.shape[0])

    # Initialize predictions.
    prediction = np.zeros(X_test.shape[0])

    # Loop over alphas and fit.
    for a in alphas:
        # Initialize a Lasso object
        lasso = Lasso(alpha=a, random_state=SEED)

        # Fit.
        lasso.fit(X, y)

        # Extract these coefficients.
        c = lasso.coef_

        # Track coefficients if we have <= 5 "non-zero" elements.
        non_zero = ~np.isclose(c, 0)
        num_non_zero = np.count_nonzero(non_zero)

        # If we're below zero, track the minimum coefficients.
        if num_non_zero <= COEFF_MAX:

            # Score.
            r2 = lasso.score(X, y)

            # If this score is the best, we'll track the coefficients,
            # and create a prediction.
            if r2 < best_score:
                # Ensure values that were close to 0 are exactly 0.
                c[~non_zero] = 0

                # Put c into coeff.
                best_coeff = c

                # Track the training score.
                best_score = r2

                # Predict.
                prediction = lasso.predict(X_test)

    return best_score, best_coeff, prediction


if __name__ == '__main__':
    ####################################################################
    # Read files
    train_df = pd.read_csv(TRAIN_FILE, encoding='cp1252').dropna(axis=0)
    test_df = pd.read_csv(TEST_FILE, encoding='cp1252').dropna(axis=0)

    # The test_df has one extra country. Line up train and test.
    test_df = test_df.loc[train_df.index]

    # Get matrices.
    train_mat = train_df.drop(['Country Name'], axis=1).values.T.astype(int)
    test_mat = test_df.drop(['Country Name'], axis=1).values.T.astype(int)

    # For now, focus on one country (column 1). Later, we'll need to
    # generalize and operate on all countries.

    # Get the number of countries.
    num_countries = train_mat.shape[1]

    # Initialize alphas.
    alphas_1 = 10 ** 10 * np.linspace(0.1, 10, 100)
    alphas_2 = 100 ** 10 * np.linspace(1.1, 10, 90)

    # Track coefficients.
    best_coeffs = pd.DataFrame(0, columns=train_df['Country Name'],
                               index=train_df['Country Name'])

    # Track training scores.
    best_scores = pd.Series(np.nan, index=train_df['Country Name'])

    # Track predictions.
    predictions = pd.DataFrame(0, columns=test_df.columns.drop('Country Name'),
                               index=test_df['Country Name'])

    # Initialize boolean for country indexing.
    other_countries = np.ones(num_countries, dtype=np.bool_)

    # Prep for the first iteration.
    other_countries[0] = False

    # TODO: Parallelize loop, if necessary.
    # Loop over all the countries (columns of the train matrix).
    for i in range(num_countries):
        country = train_df.iloc[i]['Country Name']

        # Extract X and y
        X = train_mat[:, other_countries]
        y = train_mat[:, i]
        X_test = test_mat[:, other_countries]

        # Loop over the first set of alphas, get score, coefficients,
        # and predictions.
        s, c, p = lasso_for_alphas(alphas=alphas_1, X=X, y=y, X_test=X_test)

        # If our best score came back as infinity, try again with the
        # second set of alphas.
        if s == np.inf:
            s, c, p = lasso_for_alphas(alphas=alphas_2, X=X, y=y,
                                       X_test=X_test)

        # If it happened again, better mention it.
        if s == np.inf:
            print('Failure for {}.'.format(train_df.columns.iloc[i]))

        # Assign.
        best_scores.iloc[i] = s
        best_coeffs.loc[other_countries, i] = c
        predictions.iloc[i] = p

        # Update other_countries for the next iteration.
        other_countries[i] = True
        try:
            other_countries[i+1] = False
        except IndexError:
            # We reached the end. Do nothing.
            pass

    # Save coefficients and predictions to file.
    predictions.to_csv(PRED_OUT)
    best_coeffs.to_csv(COEFF_OUT, index_label='Country')

    # Print training scores.
    # TODO: Add prediction scores.
    print('Summary of best R^2 scores:')
    best_scores.describe()
