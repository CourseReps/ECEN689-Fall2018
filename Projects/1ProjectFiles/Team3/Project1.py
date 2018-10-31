import pandas as pd
import numpy as np
import os
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from random import randint
import csv
import math

os.chdir("F:\\Code\\GitHub\\Ecen689Project1\\Python\\MB\\Project1\\Project1")

data=pd.read_csv('Data.csv', encoding='cp1252')
data.head(3)

zipCodes = []
zipCodes.append(data.iloc[0,0])
totals = []
sum = 0
for i in range(len(data.iloc[:,0])):
    if zipCodes[len(zipCodes) - 1] != data.iloc[i, 0]:
        zipCodes.append(data.iloc[i, 0])
        totals.append(sum)
        sum = 0
    sum += data.iloc[i, 1]
    if i == len(data.iloc[:,0]) - 1:
        totals.append(sum)


#For Slab Percentages
zipIndex = 0
position = 0
percentages = []
percentages.append([])
for i in range(len(data.iloc[:,0])):
    if zipCodes[zipIndex] != data.iloc[i, 0]:
        percentages.append([])
        zipIndex += 1        
    percentages[len(percentages) - 1].append(data.iloc[i,1]/totals[zipIndex])
    

#For Advance Premium Tax Credits
zipIndex = 0
position = 0
totalPerZipCode = 0
taxCredits = []
taxCredits.append([])
for i in range(len(data.iloc[:,0])):
    if zipCodes[zipIndex] != data.iloc[i, 0]:
        taxCredits[len(taxCredits) - 1].append(totalPerZipCode/totals[zipIndex])
        taxCredits.append([])
        zipIndex += 1   
        totalPerZipCode = 0     
    taxCredits[len(taxCredits) - 1].append(data.iloc[i,2]/totals[zipIndex])
    totalPerZipCode += data.iloc[i,2]
    

allData = []
for i in range(len(zipCodes)):
    combined = []
    combined.append(zipCodes[i])
    combined.append(totals[i])
    combined.extend(percentages[i])
    allData.append(combined)

with open("Output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(allData)

