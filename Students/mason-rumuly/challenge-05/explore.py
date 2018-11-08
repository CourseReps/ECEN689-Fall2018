import pandas as pd
import os
from inspect import getsourcefile
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

homeDir = os.path.sep.join(os.path.abspath(getsourcefile(lambda:0)).split(os.path.sep)[:-4]) + os.path.sep
sourceDir = homeDir + 'Challenges{0}5Files{0}'.format(os.path.sep)
targetDir = homeDir

kernels = {
    'Linear':('linear', 1),
    'Polynomial_2':('poly', 2),
    'Polynomial_3':('poly', 3),
    'Polynomial_4':('poly', 4),
    'Polynomial_5':('poly', 5),
    'Radial Basis':('rbf', 1),
    'Sigmoid':('sigmoid', 1)
}

cv_bins = 16
max_a_iter = 1000

if __name__=='__main__':

    # load data
    training = shuffle(pd.read_csv(sourceDir + '5challenge_training_masondatminer.csv', index_col=0))
    testing = pd.read_csv(sourceDir + '5challenge_testing_masondatminer.csv', index_col=0)
    # print(training.head())

    # split to train and verify sets for cross-validation
    bin_size = training.shape[0]//cv_bins
    assert bin_size > 0, 'too many bins for too few training rows'

    # cross validate each kernel
    best = [None, 0]
    for k in kernels:
        print(k)
        result = sum(cross_validate(
            SVC(
                kernel=kernels[k][0],
                gamma='scale',
                degree=kernels[k][1]
            ), 
            training[['Feature 0', 'Feature 1']], 
            training['Class'],
            cv=cv_bins,
            n_jobs=-1,
            verbose=0
        )['test_score'])/cv_bins
        if result > best[1]:
            best[0] = k
            best[1] = result
        print(result)
    print('best kernel:', best[0])

    # train best on all data
    svc = SVC(
        kernel=kernels[best[0]][0], 
        degree=kernels[best[0]][1],
        gamma='scale'
    )
    svc.fit(training[['Feature 0','Feature 1']], training['Class'])

    # plot decision regions
    x_min = min(training['Feature 0'].min(), testing['Feature 0'].min())
    x_max = max(training['Feature 0'].max(), testing['Feature 0'].max())
    y_min = min(training['Feature 1'].min(), testing['Feature 1'].min())
    y_max = max(training['Feature 1'].max(), testing['Feature 1'].max())
    density = 0.02
    xx, yy = np.meshgrid(
        np.arange(x_min-abs(x_min)*0.1, x_max+abs(x_max)*0.1, density),
        np.arange(y_min-abs(y_min)*0.1, y_max+abs(y_max)*0.1, density)
    )
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # plot
    plt.figure()
    # decision contours
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # points
    plt.scatter(training['Feature 0'], training['Feature 1'], c=training['Class'])
    plt.scatter(testing['Feature 0'], testing['Feature 1'])
    plt.show()
