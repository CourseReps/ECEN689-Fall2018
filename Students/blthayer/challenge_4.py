# Installed packages:
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLarsCV, LassoCV, RidgeCV, \
    LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import graphviz

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
PART1_INTERCEPT = os.path.join(OUT_DIR, w + '-white-intercept.csv')

PART2_PREDICTIONS = os.path.join(OUT_DIR, w + '-combined-solution.csv')

PART3_PREDICTIONS = os.path.join(OUT_DIR, w + '-red-solution.csv')

# Use consistent seed.
SEED = 1234


def part_1():
    """Linear regression on the white wine data."""
    print('*' * 79)
    print('PART 1')

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

    # Handle intercept.
    intercept = pd.DataFrame(pd.Series(out[best_model]['model'].intercept_,
                             index=['intercept']))
    intercept.T.to_csv(PART1_INTERCEPT, index=False)


def part_2():
    """Decision tree for predicting red vs. white wine.

    Some misc notes:
    - max_depth of 7 seems to be the best (varied independently)
    - Using min_samples_leaf doesn't generally appear to be helpful, and
        scikit-learn's "Tips on practical use" states 1 is often the
        best choice for classification problems.
    - PCA doesn't appear to be helpful. It also wasn't helpful when
        using the max_depth of 7.
    - Using a 'balanced' class_weight with a max_depth of 7 harmed the
        accuracy.
    - Ensuring training data had even number of red/white samples harmed
        accuracy.
    """
    print('*' * 79)
    print('PART 2')
    # Load training and testing data.
    train_df = pd.read_csv(COMBINED_TRAIN, index_col='Id')
    test_df = pd.read_csv(COMBINED_TEST, index_col='Id')

    # Split training data by classifier for feature analysis.
    # train_0 = train_df.loc[train_df['type'] == 0]
    # train_1 = train_df.loc[train_df['type'] == 1]

    # Grab X and y for easy access.
    train_size = 0.75
    test_size = 0.25
    train, test = train_test_split(train_df, train_size=train_size,
                                   test_size=test_size, random_state=SEED)

    # Get an even sampling of the two types.
    # white_train = train.loc[train['type'] == 0]
    # red_train = train.loc[train['type'] == 1]
    # white_train = white_train.sample(n=red_train.shape[0], random_state=SEED)
    # new_train = pd.concat([red_train, white_train])
    # X_train = new_train.drop(['type'], axis=1)
    # y_train = new_train['type']
    # X_test = test.drop(['type'], axis=1)
    # y_test = test['type']

    # Code below is for if we aren't ensuring even training samples.
    X_train = train.drop(['type'], axis=1)
    y_train = train['type']
    X_test = test.drop(['type'], axis=1)
    y_test = test['type']

    print('Data loaded and split {}/{} train/test.'.format(train_size,
                                                           test_size),
          end='\n')
    print('Training data has {:.1f}% red wine.'.format(np.count_nonzero(
        train['type']) / train.shape[0] * 100))
    print('Testing data has  {:.1f}% red wine.'.format(np.count_nonzero(
        test['type']) / test.shape[0] * 100), end='\n')

    # Take the naive approach of throwing the data into the API with no
    # attempts at feature reduction, etc.
    tree_basic = DecisionTreeClassifier(random_state=SEED)
    tree_basic.fit(X_train, y_train)
    tree_basic_score = tree_basic.score(X_test, y_test)
    print('Basic tree score: {:.4f}'.format(tree_basic_score))

    # Max depth of 7 may be the best.
    tree_2 = DecisionTreeClassifier(random_state=SEED, max_depth=7)
    tree_2.fit(X_train, y_train)
    tree_2_score = tree_2.score(X_test, y_test)
    print('Tree 2 score: {:.4f}'.format(tree_2_score))

    # Train tree_2 on the entire data set and write predictions.
    tree_2.fit(train_df.drop(['type'], axis=1), train_df['type'])
    pred = pd.Series(tree_2.predict(test_df), index=test_df.index,
                     name='type')
    pred.to_csv(PART2_PREDICTIONS, header=True)
    print('\nPredictions written to file.')

    # Code below was used for experimenting with tree depth, minimum
    # leaf samples, and PCA

    # print('\nLooping over tree depth:')
    # # Loop over depths until we hit our basic depth:
    # for k in range(1, tree_basic.tree_.max_depth):
    #     t = DecisionTreeClassifier(random_state=SEED, max_depth=k)
    #     t.fit(X_train, y_train)
    #     s = t.score(X_test, y_test)
    #     print('Depth {:02}, Score: {:.4f}'.format(k, s))
    #
    # print('\nLooping minimum leaf samples:')
    # for k in range(1, 100):
    #     t = DecisionTreeClassifier(random_state=SEED, min_samples_leaf=int(k))
    #     t.fit(X_train, y_train)
    #     s = t.score(X_test, y_test)
    #     print('Min leaf samples {:02}, Score: {:.4f}, Depth: {:02}'.format(
    #         int(k), s, t.tree_.max_depth))
    #
    # Perform PCA in a loop.
    # print('\nPCA scores:')
    # for f in range(1, X_train.columns.shape[0]+1):
    #     # Perform PCA with f features.
    #     p = PCA(n_components=f, whiten=True)
    #     p.fit(X_train)
    #
    #     # Fit and score the tree.
    #     t = DecisionTreeClassifier(random_state=SEED)
    #     t.fit(X=p.transform(X_train), y=y_train)
    #     s = t.score(X=p.transform(X_test), y=y_test)
    #
    #     print('Score with {:02} components: {:.4f}'.format(f, s))

    # Visualize the tree.
    # dot_data = tree.export_graphviz(tree_basic, out_file=None,
    #                                 feature_names=X_train.columns,
    #                                 class_names=['white', 'red'],
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render(view=True)


def part_3():
    """Part 3 - use white wine coefficients to predict red wine."""
    print('*' * 79)
    print('PART 2')

    # Read white wine coefficients and intercept
    coeff = pd.read_csv(PART1_COEFFICIENTS)
    intercept = pd.read_csv(PART1_INTERCEPT)

    # Read red wine testing data.
    red_test = pd.read_csv(RED_TEST, index_col='Id')

    print('All data read from file.')

    # Predict.
    predictions = intercept.values + np.matmul(red_test.values, coeff.values.T)
    print('Predictions made.')

    # Write to file.
    p = pd.DataFrame(predictions, index=red_test.index, columns=['quality'])
    p.to_csv(PART3_PREDICTIONS, header=True)
    print('Predictions written to file')


if __name__ == '__main__':
    # part_1()
    # part_2()
    part_3()