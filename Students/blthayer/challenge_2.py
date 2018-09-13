# Standard library
import struct as st
import os
import time

# Installed packages.
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd

# Define mnist files. These were decompressed with 'gzip -d <file>'
MNIST_DIR = 'mnist'
TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte'
TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte'
TEST_IMAGES_FILE = 't10k-images-idx3-ubyte'
TEST_LABELS_FILE = 't10k-labels-idx1-ubyte'

# Output directory.
OUT_DIR = './'

# This article helped read the data:
# https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1


def read_images(images_file):
    # Open file.
    with open(images_file, 'rb') as f:

        # Read magic number. Not doing anything with it.
        _ = st.unpack('>4B', f.read(4))

        # Read dimensions.
        num_imgs = st.unpack('>I', f.read(4))[0]
        num_rows = st.unpack('>I', f.read(4))[0]
        num_cols = st.unpack('>I', f.read(4))[0]

        # Each pixel is a byte:
        total_bytes = num_imgs * num_rows * num_cols

        # Read image data.
        array_dim = (num_imgs, num_rows * num_cols)
        # Ugly line ahead...
        images = (255 - np.asarray(st.unpack('>' + 'B' * total_bytes,
                                   f.read(total_bytes))).reshape(array_dim))

    return images


def read_labels(labels_file):
    # Open file.
    with open(labels_file, 'rb') as f:

        # Read magic number. Not doing anything with it.
        _ = st.unpack('>4B', f.read(4))

        # Read dimensions.
        num_labels = st.unpack('>I', f.read(4))[0]

        # Read all the labels.
        labels = np.asarray(st.unpack('>' + 'B' * num_labels,
                                      f.read(num_labels)))

    return labels

def do_lr(reduced_data, all_zero, train_labels, test_data):
    ####################################################################
    # LOGISTIC REGRESSION

    # Initialize logistic regression object. Note: newton-cg solver failed to
    # converge within 100 iterations (the default).
    lr = LR(solver='saga', multi_class='multinomial', max_iter=200, tol=0.1)

    # Time training.
    time_train_0 = time.time()

    # Fit to the reduced data.
    lr.fit(reduced_data, train_labels)

    # Report timing.
    time_train_1 = time.time()
    s = ('Logistic Regression model trained in '
         + '{:.2f} seconds.').format(time_train_1 - time_train_0)
    print(s)
    it = np.sum(lr.n_iter_)
    print('Logistic Regression training took {} iterations.'.format(it))

    # Get the coefficients.
    train_coeff = lr.coef_

    # Initialize DataFrame to hold coefficients.
    coeff = pd.DataFrame(np.zeros((num_labels, train_data.shape[1])),
                         columns=train_data.columns)

    # Place training coefficients in the coeff DataFrame.
    # TODO: Are we sure that row 0 maps to images labeled 0?
    coeff.loc[:, ~all_zero] = train_coeff

    coeff_file = OUT_DIR + '2challenge_logreg_vectors.csv'
    coeff.to_csv(coeff_file)
    print('Logistic Regression coefficients saved to {}.'.format(coeff_file))

    # Predict from the test data, and score it.
    # NOTE: We're assuming Prof. Chamberland's test data isn't
    # scrambled.

    accuracy = lr.score(test_data.loc[:, ~all_zero], test_labels)

    # Inform.
    print('Accuracy: {:.2f}'.format(accuracy))

    # Write predictions to file
    prediction_file = OUT_DIR + '2challenge_logreg.csv'
    predictions = pd.Series(lr.predict(test_data.loc[:, ~all_zero]),
                            index=np.arange(1, test_data.shape[0] + 1),
                            name='Category')
    predictions.to_csv(prediction_file, header=True,
                       index=True, index_label='Id')
    print('Predictions saved to {}'.format(prediction_file))


if __name__ == '__main__':
    # Track total program run-time.
    t0 = time.time()
    # Seed for random number generator.
    seed = 42

    ####################################################################################################################
    # READ DATA
    t_read_0 = time.time()
    # Read training images and labels from MNIST
    '''
    train_images = read_images(os.path.join(MNIST_DIR, TRAIN_IMAGES_FILE))
    train_labels = read_labels(os.path.join(MNIST_DIR, TRAIN_LABELS_FILE))

    # Read test data and labels.
    test_images = read_images(os.path.join(MNIST_DIR, TEST_IMAGES_FILE))
    '''
    test_labels = read_labels(os.path.join(MNIST_DIR, TEST_LABELS_FILE))

    # Read files from Prof. Chamberland
    train_images = pd.read_csv(os.path.join(MNIST_DIR, 'mnist_train.csv'))
    test_images = pd.read_csv(os.path.join(MNIST_DIR, 'mnist_test.csv'))

    t_read_1 = time.time()
    print('Data read in {:.2f} seconds.'.format(t_read_1 - t_read_0))

    # Extract data and labels for training set.
    train_data = train_images.drop(['Id', 'Category'], axis=1)
    train_labels = train_images['Category']

    # Extract data from testing set.
    test_data = test_images.drop(['Id'], axis=1)

    # Get the number of labels.
    num_labels = len(train_labels.unique())

    ####################################################################
    # SPARSITY
    # Exclude columns which are all 0 to reduce training time. Note that
    # there are not any other columns which have all values equal to
    # each other.
    all_zero = (train_data == 0).all()
    s = ('In the training data, there are {} columns which '
         + 'are all 0').format(np.count_nonzero(all_zero))
    print(s)

    reduced_data = train_data.loc[:, ~all_zero]

    ####################################################################
    # LOGISTIC REGRESSION
    do_lr(reduced_data, all_zero, train_labels, test_data)

    t1 = time.time()
    print('Total program runtime: {:.2f} seconds.'.format(t1 - t0))
    '''
    import matplotlib.pyplot as plt
    # Sanity check, hard-coding 28x28.
    for i in range(5):
        plt.imshow(train_images[i, :].reshape(28, 28))
        print(train_labels[i])
        plt.show(block=False)
    '''
