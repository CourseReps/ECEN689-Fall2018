# Standard library
import struct as st
from os.path import join as osp
import time

# Installed packages.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

########################################################################
# FILES

# Define mnist files. These were decompressed with 'gzip -d <file>'
MNIST_DIR = 'mnist'
TRAIN_IMAGES_FILE = osp(MNIST_DIR, 'train-images-idx3-ubyte')
TRAIN_LABELS_FILE = osp(MNIST_DIR, 'train-labels-idx1-ubyte')
TEST_IMAGES_FILE = osp(MNIST_DIR, 't10k-images-idx3-ubyte')
TEST_LABELS_FILE = osp(MNIST_DIR, 't10k-labels-idx1-ubyte')

TRAIN_FILE_CHAMBERLAND = osp(MNIST_DIR, 'mnist_train.csv')
TEST_FILE_CHAMBERLAND = osp(MNIST_DIR, 'mnist_test.csv')

# Output directory.
OUT_DIR = osp('..', '..', 'Challenges', '2Submissions', 'team1')

# Define output files.
LR_PRED_FILE = osp(OUT_DIR, '2challenge_logreg.csv')
LR_COEFF_FILE = osp(OUT_DIR, '2challenge_logreg_vectors.csv')
KNN_PRED_FILE = osp(OUT_DIR, '2challenge_knn.csv')
########################################################################

# This article helped read the MNIST data:
# https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1


def read_images(images_file):
    """Helper to read MNIST images files."""
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
    """Helper to read MNIST labels files."""
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


def do_lr(train_data, train_labels, test_data, test_labels, seed=42):
    """Helper function to perform logistic regression.

    This isn't optimally reformatted, but rather copy + pasted simply to
    get this code out of the 'main' section.
    """

    # Initialize logistic regression object.
    #
    # Note: newton-cg solver failed to converge within 100 iterations,
    # and is slow.
    #
    # Note: 'saga' is pretty fast, but 'sag' uses l2 norm for penalty.
    lr = LogisticRegression(solver='sag', multi_class='multinomial',
                            max_iter=200, tol=0.01, random_state=seed)

    # Time training.
    time_train_0 = time.time()

    # Fit to the reduced data.
    lr.fit(train_data, train_labels)

    # Report timing.
    time_train_1 = time.time()
    s = ('Logistic Regression model trained in '
         + '{:.2f} seconds.').format(time_train_1 - time_train_0)
    print(s)
    it = np.sum(lr.n_iter_)
    print('Logistic Regression training took {} iterations.'.format(it))

    # Predict from the test data, and score it.
    # NOTE: We're assuming Prof. Chamberland's test data isn't
    # scrambled.

    accuracy = lr.score(test_data, test_labels)

    # Inform.
    print('Accuracy: {:.4f}'.format(accuracy))

    # Return the trained logistic regression object.
    return lr


def map_coeff(data, labels, coeff):
    """Map coefficients from logistic regression.

    The .coef_ call doesn't give them in order, so we need to discover
    the order by checking vector performance across images."""

    # Get unique labels.
    u_labels = np.unique(labels)

    # Initialize pandas Series for tracking which coefficient vector
    # does best for each label.
    coeff_map = pd.Series(0, index=u_labels)

    # Loop over unique labels.
    for label in u_labels:
        # Extract data where this label matches.
        label_bool = labels == label

        # Extract data for this label.
        this_data = data.loc[label_bool, :]

        # Compute the dot product (via matrix multiplication) of each
        # coefficient vector with each image.
        scores = this_data.dot(coeff.transpose())

        # Run each through a pseudo objective function. Lower is better.
        scores = np.log(1 + np.exp(-scores))

        # Sum each column of scores (for each vector coefficient.
        score_sum = scores.sum()

        # Our best vector should be the minimum
        best_coeff = score_sum.idxmin()

        # Put the best coefficient in the mapping.
        coeff_map[label] = best_coeff

    # We're done. If the unique set of the coeff_map doesn't match our
    # unique labels, this failed.
    if u_labels.shape[0] != coeff_map.unique().shape[0]:
        raise UserWarning('Our coefficient mapping failed!')

    return coeff_map


def read_data():
    """Simple helper to read and return the data we need."""
    t_read_0 = time.time()
    # Read training images and labels from MNIST
    '''
    train_images = read_images(TRAIN_IMAGES_FILE)
    train_labels = read_labels(TRAIN_LABELS_FILE)

    # Read test data and labels.
    test_images = read_images(TEST_IMAGES_FILE)
    '''
    # Assume Prof. Chamberland didn't scramble the test data (turns out
    # he didn't), and read test labels directly from MNIST file.
    test_labels = read_labels(TEST_LABELS_FILE)

    # Read files from Prof. Chamberland
    train_images = pd.read_csv(TRAIN_FILE_CHAMBERLAND)
    test_images = pd.read_csv(TEST_FILE_CHAMBERLAND)

    t_read_1 = time.time()
    print('Data read in {:.2f} seconds.'.format(t_read_1 - t_read_0))

    # Extract data and labels for training set.
    train_data = train_images.drop(['Id', 'Category'], axis=1)
    train_labels = train_images['Category']

    # Extract data from testing set.
    test_data = test_images.drop(['Id'], axis=1)

    return train_data, train_labels, test_data, test_labels


def main(train_data, train_labels, test_data, test_labels, min_max_tol=0,
         write_outputs=True):
    """Perform linear regression and k-nearest neighbors for given data."""
    # Time program runtime.
    t0 = time.time()

    # Get the number of labels.
    num_labels = len(train_labels.unique())
    ####################################################################
    # FEATURE REDUCTION
    #
    # Exclude columns which have identical minimums and maximums. We
    # could take this further by using some tolerance for the delta
    # between mins and maxes.
    no_info = (train_data.max() - train_data.min()) <= min_max_tol
    s = ('In the training data, there are {} columns which do not contain '
         + 'useful information').format(np.count_nonzero(no_info))
    print(s)

    # Reduce the training and testing data.
    train_data_r = train_data.loc[:, ~no_info]
    test_data_r = test_data.loc[:, ~no_info]

    ####################################################################
    # LOGISTIC REGRESSION
    lr = do_lr(train_data_r, train_labels, test_data_r, test_labels)

    # Initialize DataFrame to hold full set of coefficients.
    coeff = pd.DataFrame(np.zeros((num_labels, train_data.shape[1])),
                         columns=train_data.columns)

    # Place training coefficients in the coeff DataFrame.
    # NOTE: These coefficients will need to be sorted.
    coeff.loc[:, ~no_info] = lr.coef_

    # Map coefficients (get them in the right order).
    coeff_map = map_coeff(train_data, train_labels, coeff)

    # Assign the categories.
    coeff['Category'] = coeff_map

    # Just in case, sort.
    coeff.sort_values(by=['Category'], inplace=True)

    # Sort by column so that Category shows up first.
    cols = list(coeff.columns)
    sorted_cols = [cols[-1]] + cols[0:-1]
    coeff = coeff[sorted_cols]

    # Write coefficients to file.
    if write_outputs:
        coeff.to_csv(LR_COEFF_FILE, index=False)
        s = 'Logistic Regression coefficients saved to {}.'.format(
            LR_COEFF_FILE)

        print(s)

        # Write predictions to file
        lr_predictions = pd.Series(lr.predict(test_data.loc[:, ~no_info]),
                                   index=np.arange(1, test_data.shape[0] + 1),
                                   name='Category')
        lr_predictions.to_csv(LR_PRED_FILE, header=True,
                              index=True, index_label='Id')
        print('Predictions saved to {}'.format(LR_PRED_FILE))

    ####################################################################
    # K NEAREST NEIGHBORS
    # Initialize KNN object.
    '''
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    # Fit with the reduced training data.
    knn.fit(train_data_r, train_labels)

    t_knn_0 = time.time()
    # Predict.
    knn_predictions = pd.Series(knn.predict(test_data_r), name='Category',
                                index=np.arange(1, test_data.shape[0] + 1))

    t_knn_1 = time.time()

    print('KNN prediction time: {:.2f}'.format(t_knn_1 - t_knn_0))

    # Score the KNN predictions.
    knn_accuracy = accuracy_score(test_labels, knn_predictions)
    print('KNN prediction accuracy: {:.4f}'.format(knn_accuracy))

    if write_outputs:
        # Write predictions to file.
        knn_predictions.to_csv(KNN_PRED_FILE, header=True, index=True,
                               index_label='Id')
        print('KNN predictions written to {}'.format(KNN_PRED_FILE))
    '''

    # Report overall run time.
    t1 = time.time()
    print('Total program runtime: {:.2f} seconds.'.format(t1 - t0))

    '''
    # Code for showing some images.
    import matplotlib.pyplot as plt
    # Sanity check, hard-coding 28x28.
    for i in range(5):
        plt.imshow(train_images[i, :].reshape(28, 28))
        print(train_labels[i])
        plt.show(block=False)
    '''


if __name__ == '__main__':
    # Read data.
    train_data, train_labels, test_data, test_labels = read_data()

    main(train_data, train_labels, test_data, test_labels,
         min_max_tol=240, write_outputs=True)

    '''
    # Loop over main with different minimum/maximum tolerances.
    for i in range(250, 256, 1):
        print('*' * 79)
        print('min_max_tol: {}'.format(i))
        main(train_data, train_labels, test_data, test_labels,
             min_max_tol=i, write_outputs=False)
    '''
