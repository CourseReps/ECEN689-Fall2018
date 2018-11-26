import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from scipy import misc

import io
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import csv
import keras
import torch
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import math
from keras.models import clone_model

pathForReading1 = "C:/Users/Droider11-PC/Documents/Kiyeob/Challenge/challenge7/all/Archive/"
fileToRead1 = "C:/Users/Droider11-PC/Documents/Kiyeob/Challenge/challenge7/all/train.csv"
df_Y = pd.read_csv(fileToRead1)
onlyfiles = [f for f in listdir(pathForReading1) if isfile(join(pathForReading1, f))]
list_X = []
list_Y = []
list_X_te = []
list_X_id = []
for file in range(len(onlyfiles)):
    filenameToRead = pathForReading1 + onlyfiles[file]
    arr = misc.imread(filenameToRead)
    if (arr.shape ==(300,400,3)) is not True:
        continue
    ID = onlyfiles[file].split('.')[0]
    idx = df_Y.index[df_Y.iloc[:,0] == int(ID)]
    if idx.empty:
        list_X_id.append(ID)
        list_X_te.append(arr)
        continue
    arr = arr.swapaxes(0,2)
    arr = arr.swapaxes(1,2)
    list_X.append(arr)
    list_Y.append(df_Y.iloc[idx[0],1])

X_data = np.array(list_X)/255
dummy = torch.tensor(X_data)
X = dummy.transpose(1,3).transpose(1,2).cpu().detach().numpy()
Y_data = [i * 1000 for i in list_Y]
# Y_data = list_Y

activation = 'relu'


model = Sequential()

model.add(Conv2D(64, (2, 2), 
        input_shape=(300, 400, 3),
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)

model.add(Conv2D(64, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)


model.add(Conv2D(256, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)


model.add(Conv2D(256, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)


model.add(Conv2D(512, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (2, 2), 
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(512,
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)

model.add(Dense(512,
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)

model.add(Dense(512,
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)

model.add(Dense(512,
        activation=activation,
        kernel_initializer='random_uniform',
        bias_initializer='zeros'
    )
)
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('linear'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.compile(optimizer="sgd", loss='mean_squared_error')
model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)
model.summary()

batchSize = 32
epochs = 200

history = model.fit(X, Y_data, batch_size=batchSize, epochs=epochs, verbose=True, validation_split=.1)

