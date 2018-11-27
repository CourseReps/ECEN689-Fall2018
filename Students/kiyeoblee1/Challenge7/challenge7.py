import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from scipy import misc

import numpy as np
import keras
import torch
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


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

model.add(Flatten())

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

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)
model.summary()

batchSize = 32
epochs = 300

history = model.fit(X, Y_data, batch_size=batchSize, epochs=epochs, verbose=True, validation_split=.1)

