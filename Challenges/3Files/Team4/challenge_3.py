########################################################################
# IMPORTS
# Standard library
from os.path import join
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
import textwrap

# Installed
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

########################################################################
# CONSTANTS
# In/out paths.
IN_DIR = join('..')
OUT_DIR = '.'

# Input files.
TEST_FILE = join(IN_DIR, 'population_testing.csv')
TRAIN_FILE = join(IN_DIR, 'population_training.csv')

# Output files.
PRED_OUT = join(OUT_DIR, 'population_prediction.csv')
COEFF_OUT = join(OUT_DIR, 'population_parameters.csv')

# Random seed.
SEED = 42

# Maximum non-zero coefficients (how many other countries we're using
# as predictors)
COEFF_MAX = 5

# Initialize alphas for Lasso.
ALL_ALPHAS = []
a = np.linspace(0.1, 0.01, 100)
b = np.logspace(1, 20, 20)
for mult in b:
    ALL_ALPHAS.append(a * mult)

# Iterations and tolerance for Lasso.
MAX_ITER = 1000000
TOL = 0.01

# Use all processors
NUM_PROCESSES = cpu_count()

# Tolerances for checking if a coefficient is close to zero. We'll go
# four orders of magnitude "closer" than the defaults.
ATOL = 1e-12
RTOL = 1e-09

# List of non-countries (aggregations) to remove so that we're only
# considering countries
NON_COUNTRIES = ['Arab World', 'Central Europe and the Baltics',
                 'Caribbean small states',
                 'East Asia & Pacific (excluding high income)',
                 'Early-demographic dividend', 'East Asia & Pacific',
                 'Europe & Central Asia (excluding high income)',
                 'Europe & Central Asia', 'Euro area', 'European Union',
                 'Fragile and conflict affected situations',
                 'High income', 'Heavily indebted poor countries (HIPC)',
                 'IBRD only', 'IDA & IBRD total', 'IDA total', 'IDA blend',
                 'IDA only',
                 'Latin America & Caribbean (excluding high income)',
                 'Latin America & Caribbean',
                 'Least developed countries: UN classification',
                 'Low income', 'Lower middle income',
                 'Low & middle income', 'Late-demographic dividend',
                 'Middle East & North Africa', 'Middle income',
                 'Middle East & North Africa (excluding high income)',
                 'North America', 'OECD members', 'Other small states',
                 'Pre-demographic dividend', 'Pacific island small states',
                 'Post-demographic dividend', 'South Asia',
                 'Sub-Saharan Africa (excluding high income)',
                 'Sub-Saharan Africa', 'Small states',
                 'East Asia & Pacific (IDA & IBRD countries)',
                 'Europe & Central Asia (IDA & IBRD countries)',
                 'Latin America & the Caribbean (IDA & IBRD countries)',
                 'Middle East & North Africa (IDA & IBRD countries)',
                 'South Asia (IDA & IBRD)',
                 'Sub-Saharan Africa (IDA & IBRD countries)',
                 'Upper middle income', 'World']

########################################################################
# FUNCTIONS


def lasso_for_alphas(alphas, x, y, x_test, y_test):
    """Loop over alphas and Lasso.

    NOTE: alphas should be in DESCENDING ORDER
    """
    # Track best score for this country.
    best_score = np.inf

    # Track the final boolean of non-zero coefficients
    final_non_zero = None

    # Loop over alphas and fit.
    for alpha in alphas:
        # Initialize a Lasso object
        lasso = Lasso(alpha=alpha, random_state=SEED, max_iter=MAX_ITER,
                      tol=TOL)

        # Fit.
        lasso.fit(x, y)

        # Extract these coefficients.
        c = lasso.coef_

        # Track coefficients if we have <= 5 "non-zero" elements.
        non_zero = ~np.isclose(c, 0, atol=ATOL, rtol=RTOL)
        num_non_zero = np.count_nonzero(non_zero)

        # If we're below zero, track the minimum coefficients.
        if (num_non_zero <= COEFF_MAX) and (num_non_zero > 0):

            # Score.
            # s = lasso.score(x, y)
            p = lasso.predict(X=x_test)
            s = mean_squared_error(y_test, p)

            # If this score is the best, we'll track the coefficients,
            # and create a prediction.
            if s < best_score:
                final_non_zero = non_zero

                # Track the training score.
                best_score = s

        elif num_non_zero >= COEFF_MAX:
            # Since we're looping in DESCENDING ORDER, we should break
            # the loop here. If the highest value of alpha gives us
            # too many coefficients, no sense in going to smaller alpha
            # values.
            break

        elif num_non_zero <= 0:
            # Our value of alpha is too high. Continue looping.
            continue

        else:
            # What's going on?
            raise UserWarning('WTF?')

    return best_score, final_non_zero


def fit_for_country(train_mat, test_mat, other_countries):
    """Loop over sets of alphas, get linear regression fit, track best.
    """
    # Extract x and y
    x = train_mat[:, other_countries]
    y = train_mat[:, ~other_countries]
    x_test = test_mat[:, other_countries]
    y_test = test_mat[:, ~other_countries]

    # Initialize returns.
    best_score = np.inf
    return_tuple = ()

    for alphas in ALL_ALPHAS:
        # Perform lasso for this set of alphas.
        score, non_zero = lasso_for_alphas(alphas=alphas, x=x, y=y,
                                           x_test=x_test, y_test=y_test)

        if not np.isinf(score):
            # Initialize linear regression object.
            lr = LinearRegression()

            # Fit and predict.
            lr.fit(x[:, non_zero], y)
            p = lr.predict(X=x_test[:, non_zero])

            # Score prediction.
            s = mean_squared_error(y_test, p)

            if s < best_score:
                # For tracking coefficients:
                c = np.zeros(np.count_nonzero(other_countries))
                # Place coefficients in the appropriate place. For some
                # reason, the coefficients come back nested (like
                # [[1, 2, 3]]), so we have to squeeze them out.
                c[non_zero] = np.squeeze(lr.coef_)
                # Update best score.
                best_score = s
                # Update final return.
                return_tuple = (s, c, p)

    # If we weren't able to get a working  alpha for this country,
    # mention it.
    if np.isinf(best_score):
        print('Failure for country {}.'.format(np.argwhere(~other_countries)))

    return return_tuple


def fit_for_country_worker(train_mat, test_mat, queue_in, queue_out):
    """Function for parallel worker"""

    while True:
        # Grab country information from the queue.
        i, num_countries = queue_in.get()

        # Initialize boolean for country indexing.
        other_countries = np.ones(num_countries, dtype=np.bool_)

        # Do not include this country.
        other_countries[i] = False

        # Leave the function if None is put in the queue.
        if other_countries is None:
            return

        # Perform the fit.
        s, c, p = fit_for_country(train_mat, test_mat, other_countries)

        # Put the data in the output queue.
        queue_out.put((other_countries, s, c, p))

        # Mark the task as complete.
        queue_in.task_done()


def fit_for_all(drop_non_countries=False):
    """Main function to perform fit for all countries."""
    ####################################################################
    # Read files
    train_df = pd.read_csv(TRAIN_FILE, encoding='cp1252',
                           index_col='Country Name').dropna(axis=0)
    test_df = pd.read_csv(TEST_FILE, encoding='cp1252',
                          index_col='Country Name').dropna(axis=0)

    # The test_df has one extra country. Line up train and test.
    test_df = test_df.loc[train_df.index]

    if drop_non_countries:
        train_df = train_df.drop(NON_COUNTRIES)
        test_df = test_df.drop(NON_COUNTRIES)

    # Get matrices.
    train_mat = train_df.values.T.astype(int)
    test_mat = test_df.values.T.astype(int)

    # Grab list and number of countries for convenience.
    countries = train_df.index.values
    num_countries = countries.shape[0]

    # Initialize queues for parallel processing.
    queue_in = JoinableQueue()
    queue_out = Queue()

    # Start processes.
    processes = []
    for i in range(NUM_PROCESSES):
        p = Process(target=fit_for_country_worker, args=(train_mat, test_mat,
                                                         queue_in, queue_out))
        p.start()
        processes.append(p)

    # Loop over all the countries (columns of the train matrix).
    for i in range(num_countries):
        # Put boolean array in the queue.
        queue_in.put((i, num_countries))

    # Wait for processing to finish.
    queue_in.join()

    # Track coefficients.
    best_coeff = pd.DataFrame(0.0, columns=countries, index=countries)

    # Track training scores.
    best_scores = pd.Series(0.0, index=countries)

    # Track predictions.
    predictions = pd.DataFrame(0.0, columns=test_df.columns, index=countries)

    # Map data.
    for _ in range(num_countries):
        # Grab data from the queue.
        other_countries, s, c, p = queue_out.get()

        country = countries[~other_countries][0]

        # Map.
        best_scores.loc[~other_countries] = s
        best_coeff.loc[other_countries, country] = c
        # p needs to be transformed (17x1 vs 1x17)
        predictions.loc[~other_countries, :] = p.T

    # Shut down processes.
    for p in processes:
        queue_in.put_nowait(None)
        p.terminate()

    predictions.transpose().to_csv(PRED_OUT, index_label='Id',
                                   encoding='cp1252')
    best_coeff.to_csv(COEFF_OUT, encoding='cp1252')

    # Print MSE
    print('Summary of MSE:')
    print(best_scores.describe())

    # # Evaluate predictions.
    # mse = pd.DataFrame(mean_squared_error(test_df.values.T,
    #                                       predictions.values.T,
    #                                       multioutput='raw_values'))
    #
    # print('Summary of prediction mean squared error:')
    # print(mse.describe())


def test_predictions():
    """Ensure the predictions written to file are close """
    # Load data
    pred = pd.read_csv(PRED_OUT, encoding='cp1252', index_col='Id')
    coef = pd.read_csv(COEFF_OUT, encoding='cp1252', index_col=0)
    actual = pd.read_csv(TEST_FILE, encoding='cp1252',
                         index_col='Country Name')

    success = True
    # Loop over countries.
    for country in pred.columns:
        # Derive predictions. Note - we'll be off by a constant (the
        # intercept term)
        derived_pred = np.matmul(coef.loc[:, country].values, actual.values)

        # Get vector of differences.
        diff = pred[country].values - derived_pred

        # If all differences are nearly the same, then we're good.
        all_same = np.all(np.isclose(diff[0], diff))

        if not all_same:
            success = False

    if success:
        print('Testing succeeded! Coefficients lead to predictions.')
    else:
        print('Testing failed! Coefficients do not lead to predictions.')

def color_bars(colors, bar_list):
    """Helper function to color bars in a bar chart."""
    # Loop and assign colors to each bar.
    color_index = 0
    for b in bar_list:
        try:
            c = colors[color_index]
        except IndexError:
            # We 'overflowed' - reset index to 0.
            color_index = 0
            c = colors[0]

        # Set bar color.
        b.set_facecolor(c)

        # Increment color index for next iteration.
        color_index += 1


def graph():
    """Function for graph/analysis portion of challenge 3.

    Creates plots for report, and saves graph as .gexf file for Gephi.
    """
    ####################################################################
    # CREATE GRAPH OF COEFFICIENTS

    # Read the coefficients file.
    coef = pd.read_csv(COEFF_OUT, encoding='cp1252', index_col=0)

    # Read the training data (we'll size the nodes based on max pop in
    # training).
    train_df = pd.read_csv(TRAIN_FILE, encoding='cp1252',
                           index_col='Country Name').dropna(axis=0)

    # Get maximum population per country.
    max_pop = train_df.max(axis=1)

    # Scale so that data is on the range [0, 1]
    node_size = (max_pop - max_pop.min()) / (max_pop.max() - max_pop.min())

    # Convert to dictionary
    node_size_dict = node_size.to_dict()

    # For networkx to use the dictionary, we need to add another
    # dictionary layer.
    node_size_dict = {x: {'size': y} for x, y in node_size_dict.items()}

    # Create lists to store from and to nodes, and coefficients
    f = []
    t = []
    c = []

    # Loop over each DataFrame column, and map non-zero coefficients
    # into f and t.
    for country in coef.columns:
        # Grab non-zero coefficients.
        non_zero = ~np.isclose(coef.loc[:, country], 0, atol=ATOL, rtol=RTOL)

        # Get weights of connected countries.
        country_weights = coef.loc[non_zero, country]
        # Get list of connected countries.
        connected_countries = coef.loc[non_zero].index.values

        # Repeat this country (from) for all our connected_countries.
        f_c = [country] * connected_countries.shape[0]

        # Add to our lists.
        f.extend(f_c)
        t.extend(connected_countries)
        c.extend(country_weights)

    # Put the from and to into a DataFrame.
    tf_df = pd.DataFrame({'from': f, 'to': t, 'weight': c})

    # Create an attribute for positive or negative correlation.
    tf_df['positive'] = tf_df['weight'] > 0

    # Make the weights positive.
    tf_df['weight'] = tf_df['weight'].abs()

    # Build a graph.
    g = nx.from_pandas_edgelist(tf_df, 'from', 'to', edge_attr=True,
                                create_using=nx.MultiDiGraph)

    # Add the node sizing as node attributes.
    nx.set_node_attributes(g, node_size_dict)

    ####################################################################
    # SAVE GRAPH TO FILE
    # Save the graph in a form we can use with Gephi.
    nx.readwrite.gexf.write_gexf(g, 'graph.gexf')

    ####################################################################
    # BAR CHART OF IN-DEGREE

    # Compute the in-degree of all nodes.
    in_degree = pd.Series(dict(g.in_degree))

    # Grab and sort degree > 0
    non_zero_degree = in_degree[in_degree[:] > 0]
    non_zero_degree.sort_values(ascending=False, inplace=True)

    # Create index for bar chart.
    ind = np.arange(1, non_zero_degree.shape[0] + 1)

    # Get figure and axes for this plot.
    fig_bar, ax_bar = plt.subplots()

    # Do initial bar plotting.
    bar_list = ax_bar.barh(ind, non_zero_degree.values)

    # Get the color cycle.
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    # Color all the bars.
    color_bars(colors, bar_list)

    # Grab labels from non_zero_degree, and wrap them.
    labels = ['\n'.join(textwrap.wrap(l, 23)) for l in
              non_zero_degree.index.values]
    # Format bar chart.
    plt.yticks(ind, labels)
    ax_bar.set_axisbelow(True)
    ax_bar.grid(b=True, which='major', axis='x')
    ax_bar.set_xlabel('In-Degree')
    plt.tight_layout()
    plt.savefig('bar.eps', type='eps', dpi=1000)

    ####################################################################
    # BOX PLOT OF COEFFICIENTS FOR NON-ZERO IN-DEGREE
    # WARNING: Variables from previous sections are re-used here.

    # First, we need to grab all the incoming weights for the non-zero
    # countries.
    weight_list = []
    for country in non_zero_degree.index:
        country_bool = tf_df['to'] == country
        # Pull weights for all incoming countries.
        country_df = tf_df.loc[country_bool, :]
        # Doing all this mumbo-jumbo to avoid a setting with copy
        # warning.
        weights = pd.Series(country_df.loc[:, 'weight'].values)
        neg_values = ~country_df.loc[:, 'positive'].values
        # Make negative weights negative.
        weights.loc[neg_values] = -1 * weights.loc[neg_values]
        weight_list.append(weights)

    # Create the box plot.
    fig_box, ax_box = plt.subplots()
    ax_box.boxplot(weight_list, vert=False, whis=[5, 95])
    plt.yticks(ind, labels)
    ax_box.set_axisbelow(True)
    ax_box.grid(b=True, which='major', axis='both')
    ax_box.set_xlabel('Coefficients')
    plt.tight_layout()
    plt.savefig('box.eps', type='eps', dpi=100)

    ####################################################################
    # ASSESS PREDICTION ERRORS
    test_df = pd.read_csv(TEST_FILE, encoding='cp1252',
                          index_col='Country Name').dropna(axis=0)

    # The test_df has one extra country. Line up train and test.
    test_df = test_df.loc[train_df.index]

    test_t = test_df.transpose()

    # Load up the predictions.
    pred_df = pd.read_csv(PRED_OUT, encoding='cp1252', index_col='Id')

    # Line up the indexes. DANGER ZONE!
    pred_df.index = test_t.index

    # Compute accuracy.
    accuracy = 1 - ((pred_df - test_t).abs()/test_t).sum()/test_t.shape[0]

    # Make boxplot of accuracy
    fig_acc, ax_acc = plt.subplots()
    ax_acc.boxplot(accuracy.values, whis=[5, 95])
    ax_acc.set_ylabel('Mean Absolute Percent Correct')
    ax_acc.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    ax_acc.set_axisbelow(True)
    ax_acc.grid(b=True, which='major', axis='y')
    plt.tight_layout()
    plt.savefig('acc_box.eps', type='eps', dpi=1000)

    # Describe.
    print('Mean prediction accuracy for all countries:')
    print(accuracy.describe())

    # Make DataFrame of accuracy and maximum population
    acc_vs_pop = pd.DataFrame({'accuracy': accuracy, 'max_pop': max_pop})
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.semilogx(acc_vs_pop['max_pop'], acc_vs_pop['accuracy'],
                        linestyle='None', marker ='o')
    ax_scatter.set_xlabel('Maximum Population, 1960-1999')
    ax_scatter.set_ylabel('Mean Prediction Accuracy')
    ax_scatter.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    plt.tight_layout()
    plt.savefig('scatter.eps', type='eps', dpi=1000)


########################################################################
# MAIN


if __name__ == '__main__':
    # Perform fit.
    fit_for_all(drop_non_countries=False)

    # Test fit.
    test_predictions()

    # Create and save graph.
    #graph()
