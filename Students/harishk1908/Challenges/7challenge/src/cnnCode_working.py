import keras
from keras import layers
import numpy as np
import glob
import os
import skimage
from keras.preprocessing import image
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

height = int(400/2)
width = int(300/2)


channels = 3

data_path = '../data/'
image_dir = data_path+'Archive/'
data_extension = '.csv'
image_extension = '.jpg'
train_file = 'train'
solution_file = 'sample'
def getData():
    filenames = glob.glob(image_dir + '*' + image_extension)
    source_images = {} 
    for filename in filenames:
        img = skimage.io.imread(filename)
        img = skimage.transform.resize(img, (height, width))
        img = img[0:width, 0:width]#, 1]
        source_images[filename[len(image_dir):-4]] = img

    def fileReader(file_name):    
        return pd.read_csv(data_path + file_name + data_extension, delimiter=',').values
    def getDataAndTargets(targetMatrix):
        data = np.zeros([len(targetMatrix), width, width, channels])
        targets = targetMatrix[:,1]
        for i in range(len(targetMatrix)):
            data[i] = source_images[str(int(targetMatrix[i,0]))]
        return data, targets

    def getDataFromFile(file_name):
        matrix = fileReader(file_name)
        return getDataAndTargets(matrix)

    x_train, y_train = getDataFromFile(train_file)
    x_test, _ = getDataFromFile(solution_file)
    plt.imshow(x_train[0])#, cmap='gray')
#    plt.show()
    plt.imshow(x_test[0])#, cmap='gray')
#    plt.show()
#    x_train = np.reshape(x_train, [len(x_train), width, width, 1]).astype('float32')
#    x_test = np.reshape(x_test, [len(x_test), width, width, 1]).astype('float32')

    return x_train.astype('float32'), y_train.astype('float32'), x_test.astype('float32')

x_train, y_train, x_test = getData()

#indicator target variables

x_train, validation_x, y_train, validation_y = train_test_split(x_train, y_train, test_size=0.0, random_state=45)
input_shape = x_train[0].shape
model = Sequential()
#model.add(MaxPooling2D(pool_size=(2, 2), input_shape = input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
#model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
#model.add(keras.layers.normalization.BatchNormalization())
model.add(Dense(1, activation='tanh'))

sgd = keras.optimizers.SGD(lr = 0.005, momentum = 0.2, decay = 0.00002, nesterov=False)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=sgd,
              )

model.summary()
#Augment training data to prevent overfitting. The training data is now harder than the testing
#data. 
datagen = ImageDataGenerator(horizontal_flip=True,
        vertical_flip=True,
        zoom_range = 0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode='nearest'
        )

dummyDatagen = ImageDataGenerator(horizontal_flip = False,
                                  vertical_flip = False,
                                  zoom_range = 0.0,
                                  width_shift_range = 0.0,
                                  height_shift_range = 0.0,
                                  fill_mode = 'nearest')
datagen.fit(x_train)
dummyDatagen.fit(x_train)
batch_size = 32
epochs = 200

num_iterations = int(len(x_train)/batch_size) + 1

model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
        steps_per_epoch=num_iterations,
        epochs=epochs,
        verbose=1,
        validation_data=(validation_x, validation_y))
"""
model.fit_generator(dummyDatagen.flow(x_train, y_train,batch_size=batch_size),
        steps_per_epoch=num_iterations,
        epochs=epochs,
        verbose=1,
        validation_data=(validation_x, validation_y))
"""
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Validation loss:', score)

testOutput = model.predict(x_test, verbose=0)

testDf = pd.read_csv(data_path + solution_file + data_extension, delimiter=',') 
testDf['DGCI'] = testOutput 
testDf.to_csv(data_path + solution_file + 'out' + data_extension, index=False)

