import pandas as pd
import numpy as np

# suppress convergence and runtime warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
# Post-challenge!
# try without affine intercept

# {
#     'lasso': {
#         'raw': {
#             'mse': 157188439779664.66,
#             'wins': 96
#         },
#         'ls': {
#             'mse': 8377815477618.837,
#             'wins': 126
#         },
#         'ridge': {
#             'mse': 8377815660989.092,
#             'wins': 108
#         }
#     },
#     'greedy': {
#         'ls': {
#             'mse': 27878478707953.973,
#             'wins': 154
#         },
#         'ridge': {
#             'mse': 275498016786900.8,
#             'wins': 107
#         }
#     },
#     'gs': {
#         'ls': {  # BEST
#             'mse': 8178598166334.731,
#             'wins': 234
#         },
#         'ridge': {
#             'mse': 8189442748006.82,
#             'wins': 207
#         }
#     }
# }

################################################################

# get error for a given setup
def mse(train, test, target, sources, regressor):
    regressor.fit(
        train.loc[sources].values.transpose(),
        train.loc[target]
    )
    return mean_squared_error(
        test.loc[target],
        regressor.predict(test.loc[sources].values.transpose())
    )

################################################################

# Trying a gram-schmidt style process of selecting basis
# Attempting to get unique information out of each match
# Alternative idea to test: just take best n out of correlation.
# Tradeoff of redundancy amount: precision vs. robustness. 
# May try hybrid system.

# settings
n_nonzero = 5 # linear combination of 5 other countries
cv_bins = 4
log10_alpha = 8  # starting value to estimate alpha
max_alpha_iter = 100

# load operative file
filename = 'Challenges/3Files/population_training.csv'
pop_data = pd.read_csv(filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)

# calc validation bin size
bin_size = (2000-1960)//cv_bins

# track progress
vis_seg = 100  # number of segments to use in display
n_steps = cv_bins*len(pop_data.index)
step = 0

# running scores (choice, regression, score statistic, value)
result = {
    'lasso': {
        'raw': {'mse': 0, 'wins': 0},
        'ls': {'mse': 0, 'wins': 0},
        'ridge': {'mse': 0, 'wins': 0}
    },
    'greedy':{
        'ls': {'mse': 0, 'wins': 0},
        'ridge': {'mse': 0, 'wins': 0}
    },
    'gs':{
        'ls': {'mse': 0, 'wins': 0},
        'ridge': {'mse': 0, 'wins': 0}
    }
}

lasso_wins = 0
greedy_wins = 0
gs_wins = 0

# cross-validate
# progress
print('Validating: [', ' '*vis_seg, ']', sep='', end='', flush=True)
for bin in range(cv_bins):
    # set aside columns for validation
    val_years = [str(yr) for yr in range(1960+(bin*bin_size), 1960+((bin+1)*bin_size))]
    light_train = pop_data.drop(val_years, axis=1)
    light_test = pop_data.loc[:, val_years]
    
    # get top-level correlation matrix of countries with each-other once
    corr_mat = light_train.transpose().corr()

    # iterate over 'countries' (some are not actually countries, but aggregates)
    for country in light_train.index:
        
        # do LASSO selection with alpha value
        tmp = light_train.drop(country)
        # narrow LASSO alpha setting for precision
        lasso_countries = []
        lasso_fit = None
        log10_amin = 0
        log10_amax = None
        for i in range(max_alpha_iter):
            lasso_fit = Lasso(alpha=10**log10_alpha, fit_intercept=False)
            lasso_fit.fit(tmp.transpose(), light_train.loc[country])

            # check result
            nz = sum(lasso_fit.coef_ != 0)
            if(nz < n_nonzero):
                log10_amax = log10_alpha
                log10_alpha = (log10_amax + log10_amin) / 2
            elif(nz > n_nonzero):
                log10_amin = log10_alpha
                if log10_amax is None:
                    log10_alpha *= 2
                else:
                    log10_alpha = (log10_amax + log10_amin) / 2
            else:
                break
            
        lasso_countries = tmp.index[
            sorted(range(len(lasso_fit.coef_)), key=lambda i: abs(lasso_fit.coef_)[i])[-n_nonzero:]
        ]

            # print(log10_amin, log10_amax, log10_alpha, nz)

        # get correlation vector for this country; drop its autocorrelation
        corr_vec = corr_mat.loc[country].drop(country)
        work_train = light_train

        # use greedy process
        greedy_countries = corr_vec.transpose().sort_values(ascending=False).head(n_nonzero).index.values

        # use gram-schmidt-like process to greedily select approximate basis
        gs_countries = []
        for a in range(max(n_nonzero - 1, 0)):
            best_match = corr_vec.abs().idxmax()
            match_vec = work_train.loc[best_match]
            gs_countries.append(best_match)
            # subtract projection into match basis
            work_train = work_train.apply(lambda row: row - match_vec * row.dot(match_vec)/match_vec.dot(match_vec), axis=1, raw=True).drop(best_match)
            # update correlation vector
            corr_vec = work_train.corrwith(work_train.loc[country], axis=1).drop(country)
        # add final matching country
        gs_countries.append(corr_vec.abs().idxmax())
        # print(country, lasso_countris, top_countries, use_countries)

        # estimate least squares error
        lasso_raw_error = mean_squared_error(light_test.loc[country], lasso_fit.predict(light_test.drop(country).T))
        lasso_ls_error = mse(light_train, light_test, country, lasso_countries, LinearRegression(fit_intercept=False))
        lasso_ridge_error = mse(light_train, light_test, country, lasso_countries, Ridge(fit_intercept=False, normalize=True))
        greedy_ls_error = mse(light_train, light_test, country, greedy_countries, LinearRegression(fit_intercept=False))
        greedy_ridge_error = mse(light_train, light_test, country, greedy_countries, Ridge(fit_intercept=False, normalize=True))
        gs_ls_error = mse(light_train, light_test, country, gs_countries, LinearRegression(fit_intercept=False))
        gs_ridge_error = mse(light_train, light_test, country, gs_countries, Ridge(fit_intercept=False, normalize=True))

        # running scores
        result['lasso']['raw']['mse'] += lasso_raw_error/n_steps
        result['lasso']['ls']['mse'] += lasso_ls_error/n_steps
        result['lasso']['ridge']['mse'] += lasso_ridge_error/n_steps
        result['greedy']['ls']['mse'] += greedy_ls_error/n_steps
        result['greedy']['ridge']['mse'] += greedy_ridge_error/n_steps
        result['gs']['ls']['mse'] += gs_ls_error/n_steps
        result['gs']['ridge']['mse'] += gs_ridge_error/n_steps
        
        i = np.argmin([
            lasso_raw_error, 
            lasso_ls_error, 
            lasso_ridge_error, 
            greedy_ls_error, 
            greedy_ridge_error, 
            gs_ls_error, 
            gs_ridge_error
        ])
        if i == 0:
            result['lasso']['raw']['wins'] += 1
        elif i == 1:
            result['lasso']['ls']['wins'] += 1
        elif i == 2:
            result['lasso']['ridge']['wins'] += 1
        elif i == 3:
            result['greedy']['ls']['wins'] += 1
        elif i == 4:
            result['greedy']['ridge']['wins'] += 1
        elif i == 5:
            result['gs']['ls']['wins'] += 1
        else:
            result['gs']['ridge']['wins'] += 1

        # progress
        step += 1
        segments = int(vis_seg*step/n_steps)
        print('\rValidating: [', 'X'*segments, ' '*(vis_seg-segments), ']', sep='', end='', flush=True)

print(' DONE')

# Result Comparison
print(result)
# print('Average Error:')
# print('LASSO', lasso_avg_error)
# print('Greed', greedy_avg_error)
# print('GramS', gs_avg_error)
# print('Wins:')
# print('LASSO', lasso_wins)
# print('Greed', greedy_wins)
# print('GramS', gs_wins)
