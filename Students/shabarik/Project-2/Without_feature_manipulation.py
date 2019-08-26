import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

new_df = pd.read_csv('weather.csv').dropna()

Y = new_df['RainToday'].values.tolist()
Y_int = np.zeros(len(Y))
Y = np.array(Y,dtype=str)
Y_int[np.where(Y=='No')[0]] = 0
Y_int[np.where(Y=='Yes')[0]] = 1
Y = to_categorical(Y_int)
Y_int = np.array(Y_int, dtype=int)
X = new_df.drop('RainToday',axis=1).values.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_int, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
sc = logreg.score(X_test,Y_test)
print('Accuracy of Logistic Regression: ', sc)

sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
sc = sgd.score(X_test, Y_test)
print('Accuracy of SGD Classifier: ', sc)

KNN = KNeighborsClassifier(n_neighbors=15)
KNN.fit(X_train,Y_train)
sc = KNN.score(X_test, Y_test)
print('Accuracy of KNN Classifier: ', sc)

svm = SVC(kernel='rbf')
svm.fit(X_train,Y_train)
sc = svm.score(X_test,Y_test)
print('Accuracy of SVM Classifier: ', sc)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
sc = random_forest.score(X_test, Y_test)
print('Accuracy of Random Forest Classifier: ', sc)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()
model.add(Dense(20, input_dim=12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile('Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=64,epochs=1500,validation_split=0.2)
sc = model.evaluate(X_test,Y_test)

print('Accuracy of Neural Networks: ', sc)
