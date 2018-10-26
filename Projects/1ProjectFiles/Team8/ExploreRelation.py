import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from RetrieveOriginalData import get_data


def scatter_with_trend(x, y, title=None, xlabel=None, ylabel=None, subplot=None):
    lr = linregress(x.reshape(-1), y)

    range_x = np.array([min(x), max(x)])
    y_hat = lr.slope*range_x + lr.intercept

    if subplot is not None:
        plt.subplot(*subplot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c='b')
    plt.plot(range_x, y_hat, c='r')

    print(lr.rvalue, lr.pvalue/2) # want single-tailed version
    
# load data
health_med = get_data()

# set up and normalize x
X = health_med['MEDHHINC15'].values.reshape((-1, 1))
X -= X.min()
X /= X.max()

plt.figure()
# strong for diabetes (|r| > 0.5)
scatter_with_trend(
    X, health_med['PCT_DIABETES_ADULTS08'],
    '2008', 'Median Income (USD)', 'Diabetes Rate (%)', (2,2,1)
)
scatter_with_trend(
    X, health_med['PCT_DIABETES_ADULTS13'], 
    '2013', 'Median Income (USD)', 'Diabetes Rate (%)', (2,2,2)
)
# moderate for obesity (0.5 > |r| > 0.3)
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
