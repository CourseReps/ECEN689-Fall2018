import pandas as pd
import os
from inspect import getsourcefile
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from SGDRegressor import SGDRegressor

homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}4Files{0}'.format(os.path.sep)
targetDir = homeDir

if __name__=='__main__':

    # load data
    white_training = pd.read_csv(sourceDir + 'winequality-white-training.csv', index_col='Id')
    white_testing = pd.read_csv(sourceDir + 'winequality-white-testing.csv', index_col='Id')
    white_prediction = pd.read_csv(sourceDir + 'winequality-white-sample.csv', index_col='Id')

    # baseline MSE achievable
    linreg = LinearRegression()
    linreg.fit(white_training.drop('quality', axis=1), white_training['quality'])
    diff = linreg.predict(white_training.drop('quality', axis=1)) - np.array(white_training['quality'])
    print(diff.dot(diff)/diff.shape[0])

    # train sgd regressor
    model = SGDRegressor(
        use_momentum=True, 
        moment_factor=0.9, 
        regularization='ridge', 
        reg_strength=0.02
    )
    learning_rates = [10**(-i)for i in range(6, 10)]
    mse_hist = [0 for _ in learning_rates]
    for i, lr in enumerate(learning_rates):
        print(lr)
        model.partial_fit(
            white_training.drop('quality', axis=1), 
            white_training['quality'], 
            learning_rate=lr, epochs=1000
        )
        mse_hist[i] = model.mse(white_training.drop('quality', axis=1), white_training['quality'])
    print(mse_hist[-1])

    plt.figure()
    plt.plot(mse_hist)
    plt.show()
