# Standard library
import struct as st
import os

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


if __name__ == '__main__':
    # Seed for random number generator.
    seed = 42

    # Read training images and labels.
    '''
    train_images = read_images(os.path.join(MNIST_DIR, TRAIN_IMAGES_FILE))
    train_labels = read_labels(os.path.join(MNIST_DIR, TRAIN_LABELS_FILE))

    # Read test data and labels.
    test_images = read_images(os.path.join(MNIST_DIR, TEST_IMAGES_FILE))
    test_labels = read_labels(os.path.join(MNIST_DIR, TEST_LABELS_FILE))
    '''

    # Read files from Prof. Chamberland
    train_images = pd.read_csv(os.path.join(MNIST_DIR, 'mnist_train.csv'))
    test_images = pd.read_csv(os.path.join(MNIST_DIR, 'mnist_test.csv'))

    train_data = train_images.drop(['Id', 'Category'], axis=1)
    train_labels = train_images['Category']

    # Initialize logistic regression object. Note: newton-cg solver failed to
    # converge within 100 iterations (the default).
    lr = LR(solver='newton-cg', multi_class='multinomial', max_iter=200)

    # Fit to the training data.
    lr.fit(train_images.iloc, train_labels.iloc)

    # Get the coefficients.
    coeff = pd.DataFrame(lr.coef_)

    coeff.to_csv('challenge2_coefficients.csv')

    # Predict from the test data, and score it.
    # mean_accuracy = test_output = lr.score(test_images, test_labels)

    # Inform.
    # print('Mean accuracy: {}'.format(mean_accuracy))

    '''
    import matplotlib.pyplot as plt
    # Sanity check, hard-coding 28x28.
    for i in range(5):
        plt.imshow(train_images[i, :].reshape(28, 28))
        print(train_labels[i])
        plt.show(block=False)
    '''
