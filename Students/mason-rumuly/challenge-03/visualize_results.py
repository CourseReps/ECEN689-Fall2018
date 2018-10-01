import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# datasource names
source_dir = 'Challenges/3Files/'
destin_dir = 'Students/mason-rumuly/challenge-03/'
train_filename = source_dir + 'population_training.csv'
test_filename = source_dir + 'population_testing.csv'
train_pred_filename = destin_dir + '3population_sanity.csv'
test_pred_filename = destin_dir + '3population_predicted.csv'

# import data
train_data = pd.read_csv(train_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)
test_data = pd.read_csv(test_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0).drop('Kuwait')
train_pred_data = pd.read_csv(train_pred_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)
test_pred_data =  pd.read_csv(test_pred_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)

# concat
real = pd.concat([train_data, test_data], axis=1, join='inner')
pred = pd.concat([train_pred_data, test_pred_data], axis=1, join='inner')

years = [int(y) for y in real.columns.values]
# graph a result
for country in real.index.values:
    plt.figure()
    plt.plot(years, real.loc[country], label='Real')
    plt.plot(years, pred.loc[country], label='Predicted')
    plt.title(country)
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.legend()
    plt.show()