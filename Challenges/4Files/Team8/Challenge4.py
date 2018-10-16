import pandas as pd
import numpy as np
import os
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from random import randint
import csv
import math

os.chdir("C:\\Users\\littl\\Code\\GitHub\\Ecen689-Challenge4\\Challenge4\\Challenge4")

def Run():
    #redTrainingDataframe = pd.read_csv('winequality-white-training.csv', encoding='cp1252').dropna(axis=0)
    redTrainingDataframe = pd.read_csv('winequality-red-training.csv', encoding='cp1252').dropna(axis=0)
    #redTrainingDataframe = pd.read_csv('winequality-combined-training.csv', encoding='cp1252').dropna(axis=0)
    redTrainingDataframe.drop(['Id'], axis=1, inplace=True)
    quality = redTrainingDataframe['quality']
    redTrainingDataframe.drop(['quality'], axis=1, inplace=True)

    return GradientDescent(redTrainingDataframe, quality, .00002)
# for white it seemed to work well with a learning rate of .00002 and 20000 to 50000 iterations

def GradientDescent(data, actual, learningRate):
    b0 = 0
    betas = np.zeros(len(data.iloc[0]))
    for i in range(20000):
        position = randint(0, len(data) - 1)
        item = data.iloc[position]
        prediction = b0 + np.dot(betas, item) 
        cost = actual[position] - prediction
        b0 = b0 + learningRate*cost
        for j in range(len(betas)):
            betas[j] = betas[j] + learningRate*data.iloc[position][j]*cost
        #if i % 1000 == 0:
        #    learningRate = learningRate * .99
            #print(learningRate)
            #print([b0].append(betas))
    return b0, betas
    
def Predict(b0, betas, features):
    return b0 + np.dot(betas, features) 

def Test(b0, betas, data, actual, position):
    print(Predict(b0, betas, data.iloc[position]))
    print(actual[position])

def MultipleTest(b0, betas, data, actual):
    for i in range(20):
        Test(b0, betas, redTrainingDataframe, quality, i)
        print(' ')

def RunTestingData(b0, betas):
    testingDataframe = pd.read_csv('winequality-white-testing.csv', encoding='cp1252').dropna(axis=0)
    testingDataframe.drop(['Id'], axis=1, inplace=True)
    quality = testingDataframe['quality']
    testingDataframe.drop(['quality'], axis=1, inplace=True)
    return MeanSquaredError(b0, betas, testingDataframe, quality)

def RootMeanSquaredError(b0, betas, data, actual):
    sum = 0
    for i in range(len(data)):
        sum += (actual[i] - Predict(b0, betas, data.iloc[i]))**2
    return math.sqrt(sum / len(data))
    
    

def Accuracy(b0, betas, data, actual):
    correct = 0
    for i in range(len(data)):
        prediction = Predict(b0, betas, data.iloc[i].values)
        print(str(prediction) + ", " + str(actual[i]))
        if prediction == actual[i]:
            correct += 1
    return correct

def Normalize(b0, betas):
    minimum = b0
    for i in range(len(betas)):
        if abs(betas[i]) < minimum:
            minimum = abs(betas[i])
    result = [b0 / minimum]
    result.append([x / minimum for x in betas])
    return result

def CreatePredictionCSVFile(predictions):
    result = []
    result.append(['Id'])
    result.append(['quality'])
    ids = np.linspace(0,len(predictions) - 1, len(predictions), dtype=int).tolist()
    result[0].extend(ids)
    result[1].extend(predictions)
    result = list(map(list, zip(*result)))
    with open("predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)

def CreateBetaFile(betas):
    result = [betas[0]]
    result.extend(betas[1].tolist())
    print(result)
    print(type(result))
    with open("betas.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(map(lambda x: [x], result))

def GeneratePredictions(b0, betas, data):
    result = []
    for i in range(len(data)):
        result.append(Predict(b0, betas, data.iloc[i].values))
    return result
    

x = Run()

#dataGlobal = pd.read_csv('winequality-white-training.csv', encoding='cp1252').dropna(axis=0)
dataGlobal = pd.read_csv('winequality-red-training.csv', encoding='cp1252').dropna(axis=0)
#dataGlobal = pd.read_csv('winequality-combined-training.csv', encoding='cp1252').dropna(axis=0)
dataGlobal.drop(['Id'], axis=1, inplace=True)
actualGlobal = dataGlobal['quality']
dataGlobal.drop(['quality'], axis=1, inplace=True)

MultipleTest(x[0], x[1], dataGlobal, actualGlobal)
RootMeanSquaredError(x[0], x[1], dataGlobal, actualGlobal)
RunTestingData(x[0], x[1])

predictions = GeneratePredictions(x[0], x[1], dataGlobal)
CreatePredictionCSVFile(predictions)

Accuracy(x[0], x[1], dataGlobal, actualGlobal)

currentBest = x
redBest = currentBest
whiteBest = currentBest

testingData = pd.read_csv('winequality-white-testing.csv', encoding='cp1252').dropna(axis=0)
testingData.drop(['Id'], axis=1, inplace=True)
predictions = GeneratePredictions(currentBest[0], currentBest[1], testingData)
CreatePredictionCSVFile(predictions)

CreateBetaFile(currentBest)
