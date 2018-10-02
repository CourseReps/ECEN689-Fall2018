import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import time
import csv
import itertools

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#alpha 5e9 -> 14 non zero coefficients
#alpha of 1.58113883e+10 -> 5 coefficients

#For all coefficient arrays, determines the number of non zero coefficients for each coefficient array 
def CheckCoefficients(allCoefficients):
    nonZeros = []
    for i in range(len(allCoefficients)):
        nonZeros.append(NumberOfNonZeros(allCoefficients[i]))
    return nonZeros

#Determines the number of nonzero values in an array
def NumberOfNonZeros(array):
    count = 0
    for i in range(len(array)):
        if array[i] != 0:
            count += 1
    return count

#For a given array of coefficients, determines which countries were non zero
def RespectiveCountries(coefficients, otherCountriesLabels):
    countries = []
    for i in range(len(coefficients)):
        if coefficients[i] != 0:
            countries.append(otherCountriesLabels[i])
    return countries

#Performs the coefficients times country population data for entire training set
def DoCalculation(otherCountries, coefficients):
    result = otherCountries[0] * 0.0
    for i in range(len(otherCountries)):
        result += otherCountries[i] * coefficients[i]
    return result

def Distance(vector1, vector2):
    return mean_squared_error(vector1, vector2)

#Does the lasso computation for a given alpha array 
def DoLasso(alphas, currentCountry, otherCountries):
    lasso = Lasso(alpha = alphas, normalize=True)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha = a)
        lasso.fit(otherCountries.transpose(), currentCountry)
        coefs.append(lasso.coef_)
    return coefs

def IsThereA5(numberOfCoefficientsPerAlhpa):
    fiveFound = False
    for i in range(len(numberOfCoefficientsPerAlhpa)):
        if numberOfCoefficientsPerAlhpa[i] == 5:
            fiveFound = True
    return fiveFound

def ApplyAlphas(alphas, currentCountry, otherCountries):
    numberOfCoefficientsPerAlpha = []
    while(not IsThereA5(numberOfCoefficientsPerAlpha)):
        coefficients = DoLasso(alphas, currentCountry, otherCountries) #Does L1 regression for every value of alpha -> returns a matrix of coefficients for each alpha
        numberOfCoefficientsPerAlpha = CheckCoefficients(coefficients) # prints out non zero count
        #print(alphas)
        #print(numberOfCoefficientsPerAlpha)
        #print()

        if(not IsThereA5(numberOfCoefficientsPerAlpha)):
            overUnderFound = False
            for i in range(len(numberOfCoefficientsPerAlpha) - 1):
                if numberOfCoefficientsPerAlpha[i] > 5 and numberOfCoefficientsPerAlpha[i + 1] < 5:
                    alphas = np.linspace(alphas[i], alphas[i + 1], linespaceCount)
                    overUnderFound = True
                    break
            if not overUnderFound:
                previousUpperBound = alphas[len(alphas) - 1]
                alphas = np.linspace(previousUpperBound, previousUpperBound * 2, linespaceCount)
    #print(numberOfCoefficientsPerAlpha)
    position = -1
    #minimum = 10000000000000000
    minimum = math.inf
    for i in range(len(numberOfCoefficientsPerAlpha)):
        if numberOfCoefficientsPerAlpha[i] == 5:
            predictedYears = DoCalculation(otherCountries, coefficients[i])
            #print(predictedYears)
            distance = Distance(currentCountry, predictedYears)
            #print(distance)
            if distance < minimum:
                minimum = distance
                position = i
    #print(minimum)
    if position == 0:
        return [alphas[position], alphas[position + 1], alphas[position]]
    elif position == len(alphas) - 1:
        return [alphas[position - 1], alphas[position], alphas[position]]
    else:
        return [alphas[position - 1], alphas[position + 1], alphas[position]]

#currentCountry -> vector of years for the currentCountry
#otherCountries -> matrix of all other country years
#def Search(currentCountry, currentCountryLabel, otherCountries, otherCountryLabels):
def Search(currentCountry, otherCountries, otherCountriesLabels):
    global linespaceCount
    linespaceCount = 25
    iterations = 3
    alphaBounds = [0, 5000]
    for i in range(iterations):
        alphas = np.linspace(alphaBounds[0], alphaBounds[1], linespaceCount)
        alphaBounds = ApplyAlphas(alphas, currentCountry, otherCountries)
        linespaceCount = 100
    chosenAlpha = alphaBounds[2]
    coefficients = DoLasso([chosenAlpha], currentCountry, otherCountries)[0]
    return RespectiveCountries(coefficients, otherCountriesLabels), coefficients

        
#iteration to loop through all the countries.
def iterationCountries(populationTrainingMatrix, trainingCountries):
    correspondingCountries = [] # the return values from search function. It contains the 5 countries
    result = [] # final result of each country with its 5 "other countries.
    coefficientResult = [] # final result of each country with its 5 "other countries.
    #searchSpace = range(0,2)
    searchSpace = range(len(populationTrainingMatrix))
    #for i in range(len(populationTrainingMatrix)):
    for i in searchSpace:
        otherCountriesNew = [] # list that carries the values of othercountries
        otherCountriesLabelsNew = [] # list of the corresponding other country label.
        trainingCountry = populationTrainingMatrix[i,:]
        j = 0
        for j in range(len(populationTrainingMatrix)):
            if j == i:
                continue
            otherCountriesNew.append(populationTrainingMatrix[j,:])
            otherCountriesLabelsNew.append(trainingCountries.iloc[j])
        correspondingCountries, coefficients = Search(trainingCountry, np.array(otherCountriesNew), np.array(otherCountriesLabelsNew))
        
        result.append([trainingCountries[i]])
        result[-1].extend(correspondingCountries)
        coefficientResult.append(coefficients)
        print(result[-1])
    return result, coefficientResult

def Run():
    populationTrainingDataframe = pd.read_csv('population_training.csv', encoding='cp1252').dropna(axis=0)

    trainingCountries = populationTrainingDataframe.iloc[:,0]
    populationTrainingDataframe.drop(['Country Name'], axis=1, inplace=True)

    searchSpace = range(len(populationTrainingDataframe))
    #searchSpace = range(0,2)

    #print("Down Search")
    #weights = np.linspace(1, .4, len(populationTrainingDataframe.iloc[0]))
    ##weights = np.linspace(.4, 1, 8)
    ##weights = list(itertools.chain.from_iterable(itertools.repeat(x, 5) for x in weights))
    #downFrame = populationTrainingDataframe.copy()
    #for i in range(len(populationTrainingDataframe)):
    #    for j in range(len(populationTrainingDataframe.iloc[i])):
    #        downFrame.iloc[i][j] = weights[j] * downFrame.iloc[i][j]

    #populationTrainingMatrix = downFrame.values
    #otherCountries = populationTrainingMatrix[1:,:]
    #otherCountriesLabels = trainingCountries.iloc[1:]
    #downResult, downCoefficientResult = iterationCountries(populationTrainingMatrix, trainingCountries)

    #print("Up Search")
    #weights = np.linspace(.4, 1, len(populationTrainingDataframe.iloc[0]))
    #upFrame = populationTrainingDataframe.copy()
    #for i in range(len(populationTrainingDataframe)):
    #    for j in range(len(populationTrainingDataframe.iloc[i])):
    #        upFrame.iloc[i][j] = weights[j] * upFrame.iloc[i][j]

    #populationTrainingMatrix = upFrame.values
    #upResult, upCoefficientResult = iterationCountries(populationTrainingMatrix, trainingCountries)

    print("None Search")
    weights = np.linspace(1, 1, len(populationTrainingDataframe.iloc[0]))
    noneFrame = populationTrainingDataframe.copy()
    for i in range(len(populationTrainingDataframe)):
        for j in range(len(populationTrainingDataframe.iloc[i])):
            noneFrame.iloc[i][j] = weights[j] * noneFrame.iloc[i][j]

    populationTrainingMatrix = noneFrame.values
    noneResult, noneCoefficientResult = iterationCountries(populationTrainingMatrix, trainingCountries)

    result = []
    coefficientResult = []
    result = noneResult
    coefficientResult = noneCoefficientResult
    #for i in searchSpace:
    #    upPredictedYears = DoCalculation(otherCountries, upCoefficientResult[i])
    #    upDistance = Distance(populationTrainingMatrix[i,:], upPredictedYears)

    #    downPredictedYears = DoCalculation(otherCountries, upCoefficientResult[i])
    #    downDistance = Distance(populationTrainingMatrix[i,:], downPredictedYears)

    #    nonePredictedYears = DoCalculation(otherCountries, noneCoefficientResult[i])
    #    noneDistance = Distance(populationTrainingMatrix[i,:], nonePredictedYears)
        
    #    downHadNegative = False
    #    for j in range(len(downPredictedYears)):
    #        if downPredictedYears[j] < 0:
    #            downHadNegative = True

    #    if downDistance < upDistance and downDistance < noneDistance and not downHadNegative:
    #        result.append(downResult[i])
    #        coefficientResult.append(downCoefficientResult[i])
    #    elif upDistance < noneDistance:
    #        result.append(upResult[i])
    #        coefficientResult.append(upCoefficientResult[i])
    #    else:
    #        result.append(noneResult[i])
    #        coefficientResult.append(noneCoefficientResult[i])

    return result, coefficientResult


def Test(coefficients):
    populationTestingDataframe = pd.read_csv('population_testing.csv', encoding='cp1252').dropna(axis=0)

    testingCountries = populationTestingDataframe.iloc[:,0]
    populationTestingDataframe.drop(['Country Name'], axis=1, inplace=True)
    populationTestingMatrix = populationTestingDataframe.values

    predictionResult = [] # final result of each country with its 5 "other countries.

    for i in range(len(populationTestingMatrix)):
        
        calculation = DoCalculation(populationTestingMatrix, coefficients[i])
        predictionResult.append(calculation)

    return predictionResult

def FixCoefficients(coefficients):
    for i in range(len(coefficients)):
        coefficients[i] = np.insert(coefficients[i], i, 0)
    return coefficients

def WriteMatchingCountriesToFile(x):
    with open("matching_countries.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(x[0])
    

def CreateCoefficientCSVFile(coefficients):
    populationTrainingDataframe = pd.read_csv('population_training.csv', encoding='cp1252').dropna(axis=0)
    trainingCountries = populationTrainingDataframe.iloc[:,0]
    result = coefficients.copy()
    for i in range(len(result)):
        result[i] = result[i].tolist()
    for i in range(len(result)):
        result[i].insert(0, trainingCountries[i])
    trainingCountries = trainingCountries.tolist()
    trainingCountries.insert(0, '')
    result.insert(0, trainingCountries)
    with open("coefficients.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)

def CreatePredictionCSVFile(predictions):
    populationTrainingDataframe = pd.read_csv('population_training.csv', encoding='cp1252').dropna(axis=0)
    trainingCountries = populationTrainingDataframe.iloc[:,0]
    result = predictions.copy()
    for i in range(len(result)):
        result[i] = result[i].tolist()
    for i in range(len(result)):
        result[i].insert(0, trainingCountries[i])
    years = np.linspace(2000, 2016, 17).tolist()
    years.insert(0, 'Country Name')
    result.insert(0, years)
    with open("predictions.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)

def MakeSureAllCoefficientsAre5(coefficients):
    all5 = True
    checkCoefficients = CheckCoefficients(coefficients)
    for i in range(len(checkCoefficients)):
        if checkCoefficients[i] != 5:
            all5 = False
    return all5

def Evaluate(testData, predictions):
    sumValue = 0
    for i in range(len(testData)):
    #for i in range(5,6):
        trueValue = testData.iloc[i, 1:]
        prediction = predictions.iloc[i, 1:]
        mse = mean_squared_error(trueValue, prediction)
        #mse = sum([abs(a - b) for a, b in zip(trueValue, prediction)])
        #print(mse)
        sumValue  += mse
    return sumValue / len(testData)

def Plot(testData, predictionX):
    for i in range(0,20):
        plt.plot(testData.columns.values[1:], testData.iloc[i,1:], color='r')
        plt.plot(predictionX.columns.values[1:], predictionX.iloc[i,1:], color='b')
        plt.show()



#os.chdir("C:\\Users\\littl\\Code\\GitHub\\Ecen689Challenge3\\Challenge3")
os.chdir("F:\\Code\\GitHub\\Ecen689Challenge3\\Challenge3")

linespaceCount = 25

x = Run()

WriteMatchingCountriesToFile(x)
selectedCoefficients = x[1].copy()
FixCoefficients(selectedCoefficients)
predictions = Test(selectedCoefficients)
CreateCoefficientCSVFile(selectedCoefficients)
CreatePredictionCSVFile(predictions)

populationTestingDataframe = pd.read_csv('population_testing.csv', encoding='cp1252').dropna(axis=0)
predictions = pd.read_csv('predictions.csv', encoding='cp1252').dropna(axis=0)
predictions1 = pd.read_csv('predictions1.csv', encoding='cp1252').dropna(axis=0)
predictions2 = pd.read_csv('predictions2.csv', encoding='cp1252').dropna(axis=0)
predictions3 = pd.read_csv('predictions3.csv', encoding='cp1252').dropna(axis=0)
predictions4 = pd.read_csv('predictions4.csv', encoding='cp1252').dropna(axis=0)
predictions5 = pd.read_csv('predictions5.csv', encoding='cp1252').dropna(axis=0)

Evaluate(populationTestingDataframe, predictions1)
Evaluate(populationTestingDataframe, predictions2)
Evaluate(populationTestingDataframe, predictions3)
Evaluate(populationTestingDataframe, predictions4)
Evaluate(populationTestingDataframe, predictions5)

Plot(populationTestingDataframe, predictions4)

y = populationTestingDataframe.iloc[i, 1:]
z = predictions1.iloc[i, 1:]
sum([abs(a - b) for a, b in zip(y,z)])
with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(x[0])

with open("coefficients.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ccoefficients)


#chosenCoefficient = coefficients[10] # need more intelligent way to get the best coefficient/alpha

#RespectiveCountries(chosenCoefficient)

#Search(trainingCounty, otherCountries, otherCountriesLabels)
