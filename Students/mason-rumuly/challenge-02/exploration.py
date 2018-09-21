import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# not working right now

# Trying a gram-schmidt style process of selecting basis
# Attempting to get unique information out of each match
# Alternative idea to test: just take best n out of correlation.
# Tradeoff of redundancy amount: precision vs. robustness. 
# May try hybrid system.

# settings
n_nonzero = 5 # linear combination of 5 other countries

# load operative file
filename = 'Challenges/3Files/population_training.csv'
pop_data = pd.read_csv(filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)

# set aside last 10 columns for validation
light_train = pop_data.drop([str(yr) for yr in range(1990, 2000)], axis=1)
light_test = pop_data.loc[:, [str(yr) for yr in range(1990, 2000)]]

# get top-level correlation matrix of countries with each-other once
corr_mat = light_train.transpose().corr()

# running scores
greedy_gs_test_difference = 0
greedy_test_wins = 0

# iterate over 'countries' (some are not actually countries, but aggregates)
for country in light_train.index:

    # get correlation vector for this country; drop its autocorrelation
    corr_vec = corr_mat.loc[country].drop(country)
    work_train = light_train

    # use greedy process
    top_countries = corr_vec.transpose().sort_values(ascending=False).head(n_nonzero).index.values

    # use gram-schmidt-like process to greedily select approximate basis
    use_countries = []
    for a in range(max(n_nonzero - 1, 0)):
        best_match = corr_vec.abs().idxmax()
        print(best_match)
        match_vec = work_train.loc[best_match]
        use_countries.append(best_match)
        # subtract projection into match basis
        work_train = work_train.apply(lambda row: row - match_vec * row.dot(match_vec)/match_vec.dot(match_vec), axis=1, raw=True).drop(best_match)
        # update correlation vector
        corr_vec = work_train.corrwith(work_train.loc[country], axis=1).drop(country)
    # add final matching country
    use_countries.append(corr_vec.abs().idxmax())
    # print(country, top_countries, use_countries)

    # estimate least squares
    ridge_gs = Ridge(normalize=True)
    ridge_gs.fit(light_train.loc[use_countries].values.transpose(), light_train.loc[country].values)
    ridge_greedy = Ridge(normalize=True)
    ridge_greedy.fit(light_train.loc[top_countries].values.transpose(), light_train.loc[country].values)
    
    # training score
    # greedy_train_score = ridge_gs.score(light_train.loc[top_countries].values.transpose(), light_train.loc[country].values)
    # greedy_train_error = mean_squared_error(ridge_gs.predict(light_train.loc[top_countries].values.transpose()), light_train.loc[country].values)
    # gs_train_score = ridge_gs.score(light_train.loc[use_countries].values.transpose(), light_train.loc[country].values)
    # gs_train_error = mean_squared_error(ridge_gs.predict(light_train.loc[use_countries].values.transpose()), light_train.loc[country].values)

    # test
    greedy_test_score = ridge_gs.score(light_test.loc[top_countries].values.transpose(), light_test.loc[country].values)
    greedy_test_error = mean_squared_error(ridge_gs.predict(light_test.loc[top_countries].values.transpose()), light_test.loc[country].values)
    gs_test_score = ridge_gs.score(light_test.loc[use_countries].values.transpose(), light_test.loc[country].values)
    gs_test_error = mean_squared_error(ridge_gs.predict(light_test.loc[use_countries].values.transpose()), light_test.loc[country].values)

    # running scores
    greedy_gs_test_difference += greedy_test_error - gs_test_error
    greedy_test_wins += greedy_test_score > gs_test_score

print('cumulative error difference:', greedy_gs_test_difference)
print('greedy wins:', greedy_test_wins)
print('gs wins:', len(light_test.index) - greedy_test_wins)