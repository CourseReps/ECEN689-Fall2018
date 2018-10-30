import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from RetrieveOriginalData import get_data, home_dir

##############################################################################################
# Functions

def cov(x, y, w):
    '''
    Weighted Covariance
    -------------------------------------------------------------------------
    x, y:  variables between which to calculate covariance 
    w:     weights
    '''
    return np.sum(
        w * (x - np.average(x, weights=w)) * (y - np.average(x, weights=w))
    ) / np.sum(w)

def corr(x, y, w):
    '''
    Weighted Pearson Correllation
    -------------------------------------------------------------------------
    x, y:  variables between which to calculate covariance 
    w:     weights
    '''
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def scatter_with_trend(x, y, w=None, title=None, xlabel=None, ylabel=None, subplot=None):
    '''
    Create a scatter plot with trendline. Returns pearson correllation, weighted if weights given
    '''
    lr = LinearRegression()
    if w is None:
        w = np.ones(y.shape)
    if len(x.shape) == 1:
        lr.fit(x.reshape((-1, 1)), y, w)
    else:
        lr.fit(x, y, w)

    range_x = np.array([min(x), max(x)]).reshape((-1, 1))
    y_hat = lr.predict(range_x)

    if subplot is not None:
        plt.subplot(*subplot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y, c='r')
    plt.plot(range_x, y_hat, c='b')

    print('Corr {} with {}: r={}'.format(xlabel, ylabel, corr(x, y, w))) # want single-tailed version

##############################################################################################
# Run
if __name__ == '__main__':

    # load data
    health_med = get_data()

    # Clean up and aggregate data

    # improve names
    health_med.rename(columns={'MEDHHINC15':'Median_Income_15'}, inplace=True)

    # Make population column and drop now-useless return columns
    health_med['Population_08'] = health_med['total_returns_08'] + health_med['joint_returns_08'] + health_med['dependents_08'] + health_med['exemptions_08']
    health_med['Population_13'] = health_med['total_returns_13'] + health_med['joint_returns_13'] + health_med['dependents_13'] + health_med['exemptions_13']

    # Create average gross income columns
    health_med['Mean_Income_kUSD_08'] = health_med['agi_kUSD_08'] / health_med['Population_08']
    health_med['Mean_Income_kUSD_13'] = health_med['agi_kUSD_13'] / health_med['Population_13']

    # Drop redundant columns
    health_med = health_med.drop([
        'total_returns_08', 'joint_returns_08', 'dependents_08', 'exemptions_08', 'agi_bracket_08', 'agi_kUSD_08',
        'total_returns_13', 'joint_returns_13', 'dependents_13', 'exemptions_13', 'agi_bracket_13', 'agi_kUSD_13'
    ], axis=1)

    # set up Xs
    XMed = health_med['Median_Income_15'].values.reshape(-1)/1000
    xmed_label = 'Median Income (kUSD, 2015)'
    X08 = health_med['Mean_Income_kUSD_08'].values.reshape(-1)
    x08_label = 'Mean Income (kUSD, 2008)'
    X13 = health_med['Mean_Income_kUSD_13'].values.reshape(-1)
    x13_label = 'Mean Income (kUSD, 2013)'

    # set up weights
    W08 = health_med['Population_08'].values.reshape(-1)
    W13 = health_med['Population_13'].values.reshape(-1)

    # check sanity of income distribution; distribution over time highly corellated
    plt.figure()
    # plt.suptitle('Income Check')
    scatter_with_trend(
        X08, X13, (W08 + W13)/2,
        'Correlation Over Time', x08_label, x13_label, (2,1,1)
    )
    scatter_with_trend(
        XMed, X08, W08,
        "Median '15 to Mean '08", xmed_label, x08_label, (2,2,3)
    )
    scatter_with_trend(
        XMed, X13, W13,
        "Median '15 to Mean '13", xmed_label, x13_label, (2,2,4)
    )
    plt.tight_layout()
    plt.savefig(home_dir + 'Visualizations/IncomeCorr.png')
    # plt.show()

    # Mean Income: medium correllation
    plt.figure()
    plt.suptitle('With Mean Income')
    scatter_with_trend(
        X08, health_med['PCT_DIABETES_ADULTS08'], W08,
        '2008', x08_label, '2008 Diabetes Rate (%)', (2,2,1)
    )
    scatter_with_trend(
        X13, health_med['PCT_DIABETES_ADULTS13'], W13,
        '2013', x13_label, '2013 Diabetes Rate (%)', (2,2,2)
    )
    scatter_with_trend(
        X08, health_med['PCT_OBESE_ADULTS08'], W08,
        None, x08_label, '2008 Obesity Rate (%)', (2,2,3)
    )
    scatter_with_trend(
        X13, health_med['PCT_OBESE_ADULTS13'], W13,
        None, x13_label, '2013 Obesity Rate (%)', (2,2,4)
    )
    plt.tight_layout()
    plt.savefig(home_dir + 'Visualizations/MeanIncomeCorr.png')
    # plt.show()

    # Median Income: Strong Correlation
    plt.figure()
    plt.suptitle('With Median Income')
    scatter_with_trend(
        XMed, health_med['PCT_DIABETES_ADULTS08'], W08,
        '2008', xmed_label, '2008 Diabetes Rate (%)', (2,2,1)
    )
    scatter_with_trend(
        XMed, health_med['PCT_DIABETES_ADULTS13'], W13,
        '2013', xmed_label, '2013 Diabetes Rate (%)', (2,2,2)
    )
    scatter_with_trend(
        XMed, health_med['PCT_OBESE_ADULTS08'], W08,
        None, xmed_label, '2008 Obesity Rate (%)', (2,2,3)
    )
    scatter_with_trend(
        XMed, health_med['PCT_OBESE_ADULTS13'], W13,
        None, xmed_label, '2013 Obesity Rate (%)', (2,2,4)
    )
    plt.tight_layout()
    plt.savefig(home_dir + 'Visualizations/MedianIncomeCorr.png')
    # plt.show()

