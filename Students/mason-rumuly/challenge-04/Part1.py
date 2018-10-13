import pandas as pd
import os
from inspect import getsourcefile
import numpy as np
from sklearn.utils import shuffle

from SGDRegressor import SGDRegressor

# best performance on latest settings, no regularization, 1000 epochs per learning rate:
# MSE Train: 0.628818335823 Test: 0.633627319819
# Score Train: 0.193460322035 Test: 0.187668296912

homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}4Files{0}'.format(os.path.sep)
targetDir = homeDir

cv_bins = 4

if __name__=='__main__':

    # load data
    white_training = shuffle(pd.read_csv(sourceDir + 'winequality-white-training.csv', index_col='Id'))
    # white_testing = pd.read_csv(sourceDir + 'winequality-white-testing.csv', index_col='Id')
    # white_prediction = pd.read_csv(sourceDir + 'winequality-white-sample.csv', index_col='Id')

    # split to train and verify sets for cross-validation
    bin_size = white_training.shape[0]//cv_bins
    assert bin_size > 0, 'too many bins for too few training rows'
    train_mse = 0
    test_mse = 0
    train_score = 0
    test_score = 0
    for bin in range(cv_bins):
        test_rows = [r for r in range(bin*bin_size, (bin+1)*bin_size)]
        light_train = white_training.drop(test_rows)
        light_test = white_training.iloc[test_rows]

        # train sgd regressor
        model = SGDRegressor(
            use_momentum=True, 
            moment_factor=0.9, 
            regularization='lasso', 
            reg_strength=0.02
        )
        learning_rates = [10**(-i)for i in range(6, 8)]
        for _ , lr in enumerate(learning_rates):
            model.partial_fit(
                light_train.drop('quality', axis=1), 
                light_train['quality'], 
                learning_rate=lr, epochs=1000
            )
        train_mse += model.mse(light_train.drop('quality', axis=1), light_train['quality'])/cv_bins
        test_mse += model.mse(light_test.drop('quality', axis=1), light_test['quality'])/cv_bins

        train_score += model.score(light_train.drop('quality', axis=1), light_train['quality'])/cv_bins
        test_score += model.score(light_test.drop('quality', axis=1), light_test['quality'])/cv_bins

    print('MSE Train:', train_mse, 'Test:', test_mse)
    print('Score Train:', train_score, 'Test:', test_score)