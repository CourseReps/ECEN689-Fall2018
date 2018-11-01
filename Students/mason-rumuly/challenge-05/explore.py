import pandas as pd
import os
from inspect import getsourcefile
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

# best performance on latest settings, no regularization, 1000 epochs per learning rate:
# MSE Train: 0.628818335823 Test: 0.633627319819
# Score Train: 0.193460322035 Test: 0.187668296912

homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}5Files{0}'.format(os.path.sep)
targetDir = homeDir

cv_bins = 4
max_a_iter = 1000

if __name__=='__main__':

    # load data
    training = shuffle(pd.read_csv(sourceDir + '5challenge_training_masondatminer.csv', index_col=0))
    # print(training.head())

    # split to train and verify sets for cross-validation
    bin_size = training.shape[0]//cv_bins
    assert bin_size > 0, 'too many bins for too few training rows'

    # plot
    plt.figure()
    plt.scatter(training.iloc[:,1], training.iloc[:,2], training.iloc[:,0])
    plt.show()
