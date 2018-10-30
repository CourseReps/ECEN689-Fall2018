import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

taken_df = pd.read_csv('2008_mapped_data.csv').dropna()

high_ppl = np.array(taken_df['AGI'].values.tolist(),dtype=int)
obese = np.array(taken_df['OBESE_ADULTS'].values.tolist(), dtype=float)
diabetes = np.array(taken_df['DIABETES_ADULTS'].values.tolist(), dtype=float)

taken_df['HIGH_RATIO'] = ((taken_df['NO_OF_HIGH_INCOME'] / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME']))*100)
taken_df['LOW_RATIO'] = (((taken_df['NO_OF_LOW_INCOME']) / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME'])) * 100)

high_ratio = np.array(taken_df['HIGH_RATIO'].values.tolist(), dtype=float)
low_ratio = np.array(taken_df['LOW_RATIO'].values.tolist(), dtype=float)

plt.scatter(high_ratio,obese)
plt.xlabel('Percentage of the population in high income group (%)')
plt.ylabel('Obesity Rate (%)')
plt.title('High Income vs Obesity in 2008')
plt.show()

plt.scatter(high_ratio,diabetes)
plt.xlabel('Percentage of the population in high income group (%)')
plt.title('High Income vs Diabetes in 2008')
plt.ylabel('Diabetes Rate (%)')
plt.show()

plt.scatter(low_ratio,obese)
plt.xlabel('Percentage of the population in low income group (%)')
plt.ylabel('Obesity Rate (%)')
plt.title('Low Income vs Obesity in 2008')
plt.show()

plt.scatter(low_ratio,diabetes)
plt.xlabel('Percentage of the population in low income group (%)')
plt.title('Low Income vs Diabetes in 2008')
plt.ylabel('Diabetes Rate (%)')
plt.show()

data = [high_ratio, obese]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in High income group and Percentage of obesity in 2008: ',corr)

data = [high_ratio, diabetes]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in High income group and Percentage of diabetes in 2008: ',corr)

data = [low_ratio, obese]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in Low income group and Percentage of obesity in 2008: ',corr)

data = [low_ratio, diabetes]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in Low income group and Percentage of diabetes in 2008: ',corr)

taken_df = pd.read_csv('2013_mapped_data.csv').dropna()

high_ppl = np.array(taken_df['AGI'].values.tolist(),dtype=int)
obese = np.array(taken_df['OBESE_ADULTS'].values.tolist(), dtype=float)
diabetes = np.array(taken_df['DIABETES_ADULTS'].values.tolist(), dtype=float)

taken_df['HIGH_RATIO'] = ((taken_df['NO_OF_HIGH_INCOME'] / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME']))*100)
taken_df['LOW_RATIO'] = (((taken_df['NO_OF_LOW_INCOME']) / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME'])) * 100)

high_ratio = np.array(taken_df['HIGH_RATIO'].values.tolist(), dtype=float)
low_ratio = np.array(taken_df['LOW_RATIO'].values.tolist(), dtype=float)

plt.scatter(high_ratio,obese)
plt.xlabel('Percentage of the population in high income group (%)')
plt.ylabel('Obesity Rate (%)')
plt.title('High Income vs Obesity in 2013')
plt.show()

plt.scatter(high_ratio,diabetes)
plt.xlabel('Percentage of the population in high income group (%)')
plt.title('High Income vs Diabetes in 2013')
plt.ylabel('Diabetes Rate (%)')
plt.show()

plt.scatter(low_ratio,obese)
plt.xlabel('Percentage of the population in low income group (%)')
plt.ylabel('Obesity Rate (%)')
plt.title('Low Income vs Obesity in 2013')
plt.show()

plt.scatter(low_ratio,diabetes)
plt.xlabel('Percentage of the population in low income group (%)')
plt.title('Low Income vs Diabetes in 2013')
plt.ylabel('Diabetes Rate (%)')
plt.show()

data = [high_ratio, obese]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in High income group and Percentage of obesity in 2013: ',corr)

data = [high_ratio, diabetes]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in High income group and Percentage of diabetes in 2013: ',corr)

data = [low_ratio, obese]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in Low income group and Percentage of obesity in 2013: ',corr)

data = [low_ratio, diabetes]

cov = np.cov(data)
num = cov[0][1]
den = sqrt(cov[0][0])*sqrt(cov[1][1])

corr = num/den

print('Correlation between Percentage of people in Low income group and Percentage of diabetes in 2013: ',corr)
