import pandas as pd
import numpy as np

# suppress convergence and runtime warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

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

# running scores
lasso_avg_error = 0
greedy_avg_error = 0
gs_avg_error = 0
lasso_wins = 0
greedy_wins = 0
gs_wins = 0

# cross-validate
# progress
print('Validating: [', ' '*vis_seg, ']', sep='', end='', flush=True)
for bin in range(cv_bins):  # 8 bins of 5 years each
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
            lasso_fit = Lasso(alpha=10**log10_alpha)
            lasso_fit.fit(tmp.transpose(), light_train.loc[country])
            abs_a = abs(lasso_fit.coef_)

            # check result
            nz = sum(abs_a != 0)
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
                lasso_countries = tmp.index[
                    sorted(range(len(abs_a)), key=lambda i: abs_a[i])[-n_nonzero:]
                ]
                break
            
            if i == (max_alpha_iter-1):
                lasso_countries = tmp.index[
                    sorted(range(len(abs_a)), key=lambda i: abs_a[i])[-n_nonzero:]
                ]

            # print(log10_amin, log10_amax, log10_alpha, nz)

        # get correlation vector for this country; drop its autocorrelation
        corr_vec = corr_mat.loc[country].drop(country)
        work_train = light_train

        # use greedy process
        top_countries = corr_vec.transpose().sort_values(ascending=False).head(n_nonzero).index.values

        # use gram-schmidt-like process to greedily select approximate basis
        use_countries = []
        for a in range(max(n_nonzero - 1, 0)):
            best_match = corr_vec.abs().idxmax()
            match_vec = work_train.loc[best_match]
            use_countries.append(best_match)
            # subtract projection into match basis
            work_train = work_train.apply(lambda row: row - match_vec * row.dot(match_vec)/match_vec.dot(match_vec), axis=1, raw=True).drop(best_match)
            # update correlation vector
            corr_vec = work_train.corrwith(work_train.loc[country], axis=1).drop(country)
        # add final matching country
        use_countries.append(corr_vec.abs().idxmax())
        # print(country, lasso_countris, top_countries, use_countries)

        # estimate least squares
        # ridge_lasso = Ridge(normalize=True)
        # ridge_lasso.fit(light_train.loc[lasso_countries].values.transpose(), light_train.loc[country].values)
        ridge_gs = LinearRegression()  # Ridge(normalize=False)
        ridge_gs.fit(light_train.loc[use_countries].values.transpose(), light_train.loc[country].values)
        ridge_greedy = Ridge(normalize=True)
        ridge_greedy.fit(light_train.loc[top_countries].values.transpose(), light_train.loc[country].values)

        # training score
        # greedy_train_score = ridge_greedy.score(light_train.loc[top_countries].values.transpose(), light_train.loc[country].values)
        # greedy_train_error = mean_squared_error(ridge_greedy.predict(light_train.loc[top_countries].values.transpose()), light_train.loc[country].values)
        # gs_train_score = ridge_gs.score(light_train.loc[use_countries].values.transpose(), light_train.loc[country].values)
        # gs_train_error = mean_squared_error(ridge_gs.predict(light_train.loc[use_countries].values.transpose()), light_train.loc[country].values)

        # test
        # lasso_test_score = ridge_lasso.score(light_test.loc[lasso_countries].values.transpose(), light_test.loc[country].values)
        # lasso_test_error = mean_squared_error(ridge_lasso.predict(light_test.loc[lasso_countries].values.transpose()), light_test.loc[country].values)
        lasso_test_error = mean_squared_error(lasso_fit.predict(light_test.drop(country).transpose()), light_test.loc[country].values)
        # greedy_test_score = ridge_greedy.score(light_test.loc[top_countries].values.transpose(), light_test.loc[country].values)
        greedy_test_error = mean_squared_error(ridge_greedy.predict(light_test.loc[top_countries].values.transpose()), light_test.loc[country].values)
        # gs_test_score = ridge_gs.score(light_test.loc[use_countries].values.transpose(), light_test.loc[country].values)
        gs_test_error = mean_squared_error(ridge_gs.predict(light_test.loc[use_countries].values.transpose()), light_test.loc[country].values)

        # running scores
        lasso_avg_error += lasso_test_error/n_steps
        greedy_avg_error += greedy_test_error/n_steps
        gs_avg_error += gs_test_error/n_steps
        
        i = np.argmin([lasso_test_error, greedy_test_error, gs_test_error])
        if i == 0:
            lasso_wins += 1
        elif i == 1:
            greedy_wins += 1
        elif i == 2:
            gs_wins += 1

        # progress
        step += 1
        segments = int(vis_seg*step/n_steps)
        print('\rValidating: [', 'X'*segments, ' '*(vis_seg-segments), ']', sep='', end='', flush=True)

print(' DONE')

# Result Comparison
print('Average Error:')
print('LASSO', lasso_avg_error)
print('Greed', greedy_avg_error)
print('GramS', gs_avg_error)
print('Wins:')
print('LASSO', lasso_wins)
print('Greed', greedy_wins)
print('GramS', gs_wins)
