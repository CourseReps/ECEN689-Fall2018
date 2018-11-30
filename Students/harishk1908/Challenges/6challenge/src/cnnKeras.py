import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
data_path = '../data/'
data_extension = '.csv'
train_file = 'mnist_train'
test_file = 'mnist_test'
solution_file = 'mnist_sample'

def fileReader(file_name):    
    return pd.read_csv(data_path + file_name + data_extension, delimiter=',').values[:,1:]

def getDataAndTargets(matrix):
    return matrix[:, 1:], matrix[:, 0]

def getData(matrix):
    return matrix

def getDataFromFile(file_name, shouldGetTargets = 1):
    matrix = fileReader(file_name)
    if(shouldGetTargets):
        return getDataAndTargets(matrix)
    return getData(matrix)

img_rows, img_cols = 28, 28

trainX, trainY = getDataFromFile(train_file)
trainX = np.reshape(trainX, [len(trainX), img_rows, img_cols])
#plt.imshow(trainX[0], cmap='gray')
#plt.show()
testX = getDataFromFile(test_file, 0)
testX = np.reshape(testX, [len(testX), img_rows, img_cols])
#plt.imshow(testX[0], cmap='gray')
#plt.show()
batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
trainX = np.reshape(trainX, [len(trainX), img_rows, img_cols, 1]).astype('float32')/255.0
testX = np.reshape(testX, [len(testX), img_rows, img_cols, 1]).astype('float32')/255.0

#indicator target variables
trainY = keras.utils.to_categorical(trainY, num_classes)

trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=0.1, random_state=42)
input_shape = trainX[0].shape
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

sgd = keras.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.00025, nesterov=False)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

#Augment training data to prevent overfitting. The training data is now harder than the testing
#data. 
datagen = ImageDataGenerator(horizontal_flip=False,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode='nearest'
        )
datagen = ImageDataGenerator(
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(trainX)

num_iterations = int(len(trainX)/batch_size) + 1

model.fit_generator(datagen.flow(trainX, trainY,batch_size=batch_size),
        steps_per_epoch=num_iterations,
        epochs=epochs,
        verbose=1,
        validation_data=(validationX, validationY))

score = model.evaluate(validationX, validationY, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

testIndicatorOut = model.predict(testX, verbose=0)
testClasses = np.argmax(testIndicatorOut, axis=1)


testDf = pd.read_csv(data_path + solution_file + data_extension, delimiter=',') 
testDf['Category'] = testClasses.astype('int') 
testDf.to_csv(data_path + solution_file + 'out' + data_extension, index=False)

