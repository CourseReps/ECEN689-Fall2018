import pandas as pd
import numpy as np


def categorical_numerical(data):
  uniq = list(np.unique(data))
  ret_data = []
  for row in data:
    tmp = uniq.index(row)
    ret_data.append(tmp)
  return ret_data


df = pd.read_csv('weatherAUS.csv')
df.head(3)

loc = df['Location'].values.tolist()
loc = categorical_numerical(loc)

new_df = pd.DataFrame()
new_df['Location'] = loc
new_df['MinTemp'] = df['MinTemp']
new_df['MaxTemp'] = df['MaxTemp']
new_df['WindGustSpeed'] = df['WindGustSpeed']
new_df['WindSpeed9am'] = df['WindSpeed9am']
new_df['WindSpeed3pm'] = df['WindSpeed3pm']
new_df['Humidity9am'] = df['Humidity9am']
new_df['Humidity3pm'] = df['Humidity3pm']
new_df['Pressure9am'] = df['Pressure9am']
new_df['Pressure3pm'] = df['Pressure3pm']
new_df['Temp9am'] = df['Temp9am']
new_df['Temp3pm'] = df['Temp3pm']
new_df['RainToday'] = df['RainToday']

new_df.to_csv('weather.csv',index=False)
