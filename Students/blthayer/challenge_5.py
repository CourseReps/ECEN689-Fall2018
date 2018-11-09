"""Module for Challenge 5 (Support Vector Machines)"""
########################################################################
# IMPORTS

# Third-party installed
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# Standard library
import os.path as osp

plt.interactive(False)
########################################################################
# CONSTANTS

# Texas A&M network ID.
NET_ID = 'blthayer'

# Directory for challenge 5 files.
DIR = osp.join('..', '..', 'Challenges', '5Files')
TRAIN_FILE = osp.join(DIR, '5challenge_training_' + NET_ID + '.csv')
TEST_FILE = osp.join(DIR, '5challenge_testing_' + NET_ID + '.csv')

# Color map:
CMAP = plt.cm.Set2

########################################################################
# FUNCTIONS BORROWED FROM SCIKIT-LEARN (AND MODIFIED):
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = x.min() * 1.5, x.max() * 1.5
    y_min, y_max = y.min() * 1.5, y.max() * 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

########################################################################
# FUNCTIONS


def plot_svm(x, y, x_test, svm_obj, kernel_str):
    """Helper function for plotting SVM predictions"""
    # Initialize figure.
    fig, ax = plt.subplots(1, 1)

    # Make a mesh grid.
    xx, yy = make_meshgrid(x[:, 0], x[:, 1])

    # Plot the decision boundaries.
    plot_contours(ax, svm_obj, xx, yy, cmap=CMAP, alpha=0.8)

    ax.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25,
               edgecolor='k')
    ax.plot(x_test[:, 0], x_test[:, 1], marker='x', c='k', linestyle='None',
            markersize=5)
    # ax.set_title('Training predictions - {}'.format(kernel_str))
    ax.set_xlabel('Feature 0 (normalized)')
    ax.set_ylabel('Feature 1 (normalized)')

    # plt.show()
    return fig, ax


def main():
    """Main function for challenge 5."""
    # Read training and testing data.
    df_train = pd.read_csv(TRAIN_FILE, index_col=0)
    df_test = pd.read_csv(TEST_FILE, index_col=0)

    # print('Training data:')
    # print(df_train.head())
    #
    # print('Testing data:')
    # print(df_test.head())

    # Extract data from DataFrame for ease of use.
    x_train = df_train[['Feature 0', 'Feature 1']].values
    y_train = df_train[['Class']].values.ravel()
    x_test = df_test[['Feature 0', 'Feature 1']].values

    # Scale data before using SVM.
    # NOTE: All kernels except the polynomial (degree 2, coef0 0) did
    # better when data is scaled to (-1, 1) as opposed to (0, 1)
    scaler = MinMaxScaler((-1, 1))
    scaler.fit(np.vstack((x_train, x_test)))

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Plot scaled training data.
    fig_train, ax_train = plt.subplots(1, 1)
    ax_train.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], marker='o',
                     c=y_train, s=25, edgecolor='k', cmap=CMAP)
    # ax_train.set_title('Training Data')
    ax_train.set_xlabel('Feature 0 (normalized)')
    ax_train.set_ylabel('Feature 1 (normalized)')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('c5_training_data.eps', type='eps')

    # Add testing data.
    ax_train.plot(x_test_scaled[:, 0], x_test_scaled[:, 1], marker='x',
                  linestyle='None', markersize=5, color='k')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('c5_train_test_data.eps', type='eps')

    # Plot testing data (we don't have y).
    # fig_test, ax_test = plt.subplots(1, 1)
    # ax_test.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], marker='o', s=25,
    #                 edgecolor='k')
    # ax_test.set_title('Testing Data')

    # Linear SVM
    # 80.0% accurate
    # svm_linear = SVC(kernel='linear')
    # svm_linear.fit(x_train_scaled, y_train)
    # acc_linear = svm_linear.score(x_train_scaled, y_train)
    # print('Linear kernel: {:.2f}% accurate'.format(acc_linear*100))
    # plot_svm(x_train_scaled, y_train, svm_linear, 'linear')


    # Polynomial, degree 2, coef0 0
    # 52.0% accurate
    # svm_poly2 = SVC(kernel='poly', degree=2, coef0=0)
    # svm_poly2.fit(x_train_scaled, y_train)
    # acc_poly2 = svm_poly2.score(x_train_scaled, y_train)
    # print('Polynomial (2) kernel: {:.2f}% accurate'.format(acc_poly2*100))
    # plot_svm(x_train_scaled, y_train, svm_poly2, 'poly2')

    # Polynomial, degree 3, coef0 0
    # 54% accurate
    # svm_poly3 = SVC(kernel='poly', degree=3, coef0=0)
    # svm_poly3.fit(x_train_scaled, y_train)
    # acc_poly3 = svm_poly3.score(x_train_scaled, y_train)
    # print('Polynomial (3) kernel: {:.2f}% accurate'.format(acc_poly3*100))
    # plot_svm(x_train_scaled, y_train, svm_poly3, 'poly3')

    # rbf, gamma=1
    # 84.0% accurate
    # svm_rbf1 = SVC(kernel='rbf', gamma=1)
    # svm_rbf1.fit(x_train_scaled, y_train)
    # acc_rbf1 = svm_rbf1.score(x_train_scaled, y_train)
    # print('RBF kernel: {:.2f}% accurate'.format(acc_rbf1*100))
    # plot_svm(x_train_scaled, y_train, x_test_scaled, svm_rbf1,
    #          'rbf, $\gamma = 1$')

    # rbf, gamma=10
    #
    # svm_rbf10 = SVC(kernel='rbf', gamma=10)
    # svm_rbf10.fit(x_train_scaled, y_train)
    # acc_rbf10 = svm_rbf10.score(x_train_scaled, y_train)
    # print('RBF kernel: {:.2f}% accurate'.format(acc_rbf10*100))
    # plot_svm(x_train_scaled, y_train, x_test_scaled, svm_rbf10,
    #          '$\gamma = 10$')

    svm_rbf10 = SVC(C=1, kernel='rbf', gamma=10)
    svm_rbf10.fit(x_train_scaled, y_train)
    acc_rbf10 = svm_rbf10.score(x_train_scaled, y_train)
    print('RBF kernel: {:.2f}% accurate'.format(acc_rbf10*100))
    fig_rbf, ax_rbf = plot_svm(x_train_scaled, y_train, x_test_scaled,
                               svm_rbf10, '$\gamma = 100$')
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('c5_decision_region.eps')

    # rbf, gamma=0.1
    #
    # svm_rbf0 = SVC(kernel='rbf', gamma=0.1)
    # svm_rbf0.fit(x_train_scaled, y_train)
    # acc_rbf0 = svm_rbf0.score(x_train_scaled, y_train)
    # print('RBF kernel: {:.2f}% accurate'.format(acc_rbf0*100))
    # plot_svm(x_train_scaled, y_train, x_test_scaled, svm_rbf0,
    #          '$\gamma = 0.1$')

    # Grid search.
    param_range = np.logspace(-4, 4, 9)
    param_grid = {'C': param_range, 'gamma': param_range}
    # It would seem scikit is smart enough to automagically select rbf.
    # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
    grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=-1)
    grid.fit(x_train_scaled, y_train)
    print("The best parameters are {} with a score of "
          "{:.2f}".format(grid.best_params_, grid.best_score_))

    # Train with the best params.
    svm_best = SVC(kernel='rbf', **grid.best_params_)
    svm_best.fit(x_train_scaled, y_train)
    s = '$\gamma = {}, C = {}$'.format(grid.best_params_['gamma'],
                                        grid.best_params_['C'])
    fig_grid, ax_grid = plot_svm(x_train_scaled, y_train, x_test_scaled,
                                 svm_best, s)
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('c5_grid_region.eps')

    # sigmoid, r = 0
    # 79.5% accurate
    # svm_sigmoid0 = SVC(kernel='sigmoid', coef0=0)
    # svm_sigmoid0.fit(x_train_scaled, y_train)
    # acc_sigmoid0 = svm_sigmoid0.score(x_train_scaled, y_train)
    # print('Sigmoid kernel: {:.2f}% accurate'.format(acc_sigmoid0*100))
    # plot_svm(x_train_scaled, y_train, svm_sigmoid0, 'sigmoid')

    # Perform prediction with testing data, write file.
    y_test = svm_rbf10.predict(x_test)
    df_test['Class'] = y_test
    df_test.to_csv(TEST_FILE)

########################################################################
# MAIN


if __name__ == '__main__':
    main()
    # plt.show(block=True)
