import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

taken_df = pd.read_csv('2013_mapped_data.csv').dropna()

obese = np.array(taken_df['OBESE_ADULTS'].values.tolist(), dtype=float)
obese = obese[:,np.newaxis]
diabetes = np.array(taken_df['DIABETES_ADULTS'].values.tolist(), dtype=float)
diabetes = diabetes[:,np.newaxis]

taken_df['HIGH_RATIO'] = ((taken_df['NO_OF_HIGH_INCOME'] / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME']))*100)
taken_df['LOW_RATIO'] = (((taken_df['NO_OF_LOW_INCOME']) / (taken_df['NO_OF_DEPENDENTS'] + taken_df['NO_OF_LOW_INCOME'] + taken_df['NO_OF_MIDDLE_INCOME'] + taken_df['NO_OF_HIGH_INCOME'])) * 100)

high_ratio = np.array(taken_df['HIGH_RATIO'].values.tolist(), dtype=float)
print(np.shape(high_ratio))
high_ratio = high_ratio[:,np.newaxis]
print(np.shape(high_ratio))
low_ratio = np.array(taken_df['LOW_RATIO'].values.tolist(), dtype=float)
low_ratio = low_ratio[:,np.newaxis]

lin1 = LinearRegression()
lin1.fit(high_ratio,diabetes)

plt.scatter(high_ratio,diabetes)
plt.xlabel('Percentage of the population in high income group (%)')
plt.ylabel('Diabetes Rate (%)')
plt.title('Linear Model on High Income vs Diabetes in 2013')
plt.plot(high_ratio,lin1.predict(high_ratio), color='k')
plt.show()
