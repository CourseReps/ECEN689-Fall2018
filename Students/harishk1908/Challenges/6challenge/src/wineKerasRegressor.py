import pandas as pd
import numpy as np
import keras
from keras.optimizers import SGD, adadelta, Adagrad
from keras.models import Sequential
from keras.layers import Dense, Dropout, GaussianNoise
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data_path = '../data/'
data_extension = '.csv'
train_file = 'winequality-white-training'
test_file = 'winequality-white-testing'
solution_file = 'winequality-white-sample'

epochs = 500

def fileReader(file_name):
    return pd.read_csv(data_path + file_name + data_extension, delimiter=',').drop(['density'], axis=1).values[:,1:]

def getDataAndTargets(matrix):
    return matrix[:, :-1], matrix[:, -1]

def getData(matrix):
    return matrix

def getDataFromFile(file_name, shouldGetTargets = 1):
    matrix = fileReader(file_name)
    if(shouldGetTargets):
        return getDataAndTargets(matrix)
    return getData(matrix)

trainX, trainY = getDataFromFile(train_file, 1)
testX = getDataFromFile(test_file,0)

dataScaler = preprocessing.StandardScaler().fit(trainX)
trainX = dataScaler.transform(trainX)
testX = dataScaler.transform(testX)

trainY = trainY#/9.0
input_shape = trainX[0].shape
model = Sequential()

model.add(GaussianNoise(0.05, input_shape=input_shape))
model.add(Dense(30, activation='relu', input_shape=input_shape))
#model.add(Dense(15, activation='relu'))
#model.add(Dense(10, activation='relu', input_shape=input_shape))
#model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))

model.add(Dense(1, activation='linear', input_shape=input_shape))

sgd = SGD(lr=0.003, decay=0.000004, momentum=0.3, nesterov=False)
adagrad = Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=sgd)
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.8, 
                                            min_lr=0.0001)
print(model.summary())
trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=0.1, random_state=40)

model.fit(trainX, trainY, verbose=1, epochs=epochs,batch_size=128, validation_data=(validationX, validationY))#, callbacks=[learning_rate_reduction])



testOutput = model.predict(testX, verbose=0)

testDf = pd.read_csv(data_path + solution_file + data_extension, delimiter=',')
testDf['quality'] = testOutput
testDf.to_csv(data_path + solution_file + 'out' + data_extension, index=False)

