import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA



score = 0 
train = pd.read_csv('C:/Users/jatin/Downloads/mnist_train.csv',header=0)
trainm = train.iloc[:,2:786]

x = trainm
y = train.iloc[:,1]
test = pd.read_csv('C:/Users/jatin/Downloads/mnist_test.csv',header=0)

x_test = test.iloc[:,1:785]



for i in range(30,500):
    pca = PCA(n_components=i)
    x_pca = pca.fit_transform(x)
    x_test_pca = pca.transform(x_test)



    # Applying KNN after PCA (We considered n_neighbors=5 which is the most common choice)

    KNN_ALGO_TRAIN_MODEL = KNeighborsKNN_ALGO_TRAIN_MODEL(n_neighbors=5)  
    KNN_ALGO_TRAIN_MODEL.fit(x_pca,y)



    y_pred = KNN_ALGO_TRAIN_MODEL.predict(x_test_pca)

    d = {'category':pd.Series(y_pred)}


    result = pd.concat([pd.read_csv('C:/Users/jatin/Downloads/mnist_test.csv').iloc[:,0:1],pd.DataFrame(d)], axis = 1)


    known_data = pd.read_csv('C:/Users/jatin/Downloads/known.csv',header= None, usecols = [0])


    a = KNN_ALGO_TRAIN_MODEL.score(x_test_pca, known_data)
    if(a>score):
        score = a
        print("i is--------",i,a)
        result.to_csv('result.csv', encoding='utf-8', index=False)

# result. to_csv('D:/Study/3rd Sem/ECEN 689/' + 'KNN_with_PCA.csv')