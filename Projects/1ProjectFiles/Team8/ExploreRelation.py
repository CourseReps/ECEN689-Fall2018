import matplotlib.pyplot as plt
from RetrieveOriginalData import get_data
from sklearn.linear_model import LinearRegression
import numpy as np

def scatter_with_trend(x, y, title=None, xlabel=None, ylabel=None, subplot=None):
    lr = LinearRegression()
    lr.fit(x, y)

    range_x = [min(x), max(x)] 

    if subplot is not None:
        plt.subplot(*subplot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c='b')
    plt.plot(range_x, lr.predict(range_x), c='r')
    
# load data
health_med = get_data()

# set up and normalize x
X = health_med['MEDHHINC15'].values.reshape((-1, 1))
X -= X.min()
X /= X.max()

plt.figure()
scatter_with_trend(
    X, health_med['PCT_DIABETES_ADULTS08'],
    '2008', 'Median Income (USD)', 'Diabetes Rate (%)', (2,2,1)
)
scatter_with_trend(
    X, health_med['PCT_DIABETES_ADULTS13'], 
    '2013', 'Median Income (USD)', 'Diabetes Rate (%)', (2,2,2)
)
scatter_with_trend(
    X, health_med['PCT_OBESE_ADULTS08'],
    None, 'Median Income (USD)', 'Obesity Rate (%)', (2,2,3)
)
scatter_with_trend(
    X, health_med['PCT_OBESE_ADULTS13'], 
    '2013', 'Median Income (USD)', 'Obesity Rate (%)', (2,2,4)
)
plt.tight_layout()
plt.show()