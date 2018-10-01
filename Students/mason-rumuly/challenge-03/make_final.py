import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


################################################################
# helper functions

def mode(iterable):
    '''get mode of list. If multiple, picks first'''
    names = []
    count = []
    for value in iterable:
        if value not in names:
            names.append(value)
            count.append(1)
        else:
            count[names.index(value)] += 1
    return(names[count.index(max(count))])


def to_sparse_vector(chosen_countries, coef, all_countries):
    '''makes sparse list with coefficients in order. Fills in zeros'''
    return [
        0 if country not in chosen_countries else coef[chosen_countries.index(country)] 
        for country in all_countries
    ]


################################################################
# country choice functions

# mse over test set: 5.30564383207e+13
def lasso_choose_countries(target, n, dataframe, log10_alpha = 8, max_alpha_iter = 1000):
    '''
    Chooses n countries from the dataframe which best predict the target country value
    Uses a Lasso process which uses binary estimation to find a correct alpha value
    '''
    # temp to avoid messing up original
    tmp = dataframe.drop(target)
    # normalize the rows to avoid over-emphasizing large populations
    tmp = (tmp - tmp.mean()) / (tmp.max() - tmp.min())
    # narrow LASSO alpha setting for precision
    log10_amin = 0
    log10_amax = None
    abs_a = []
    for _ in range(max_alpha_iter):
        lasso_fit = Lasso(alpha=10**log10_alpha)
        lasso_fit.fit(tmp.transpose(), dataframe.loc[target])
        abs_a = abs(lasso_fit.coef_)

        # check result
        nz = sum(abs_a != 0)
        if(nz < n):
            log10_amax = log10_alpha
            log10_alpha = (log10_amax + log10_amin) / 2
        elif(nz > n):
            log10_amin = log10_alpha
            if log10_amax is None:
                log10_alpha *= 2
            else:
                log10_alpha = (log10_amax + log10_amin) / 2
        else:  # converged
            break
    # return choice
    # print(log10_alpha)
    return tmp.index[
        sorted(range(len(abs_a)), key=lambda i: abs_a[i])[-n:]
    ]

# mse over test set: 5.48543938289e+13
def gs_choose_countries(target, n, dataframe):
    '''
    Chooses n countries from the dataframe which best predict the target country value
    Uses a process based on Grahm-Schmidt orthonormal base derivation to get as many
    different parts of information as possible
    '''
    # Work iteratively, each choice influenced by others
    work_train = dataframe
    use_countries = []
    for _ in range(n_nonzero):
        # get correllation vector
        corr_vec = work_train.corrwith(work_train.loc[target], axis=1).drop(target).dropna()
        if corr_vec.empty:
            break
        # get best correlation
        best_match = corr_vec.abs().idxmax(skipna=False)
        match_vec = work_train.loc[best_match]
        use_countries.append(best_match)
        # subtract projection into match basis
        work_train = work_train.apply(lambda row: row - match_vec * row.dot(match_vec)/match_vec.dot(match_vec), axis=1, raw=True).drop(best_match)
    return use_countries

################################################################
# run
if __name__ == "__main__":

    # settings
    n_nonzero = 5 # linear combination of 5 other countries
    submission_loc = 'Students/mason-rumuly/challenge-03/'

    # load operative files
    train_filename = 'Challenges/3Files/population_training.csv'
    pop_data = pd.read_csv(train_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0)
    test_filename = 'Challenges/3Files/population_testing.csv'
    pop_test = pd.read_csv(test_filename, index_col='Country Name', encoding='cp1252').dropna(axis=0).drop('Kuwait')

    # for each country
    countries = pop_data.index.values
    chosen_countries = [lasso_choose_countries(country, n_nonzero, pop_data) for country in countries]
    regressions = [LinearRegression() for _ in countries]
    regressions = [
        regressions[i].fit(pop_data.loc[chosen_countries[i]].values.transpose(), pop_data.loc[country].values) 
        for i, country in enumerate(countries)
    ]
    predictions = [
        regressions[i].predict(pop_test.loc[chosen_countries[i]].values.transpose()) 
        for i in range(len(countries))
    ]
    mse = [
        mean_squared_error(predictions[i], pop_test.loc[country].values)
        for i, country in enumerate(countries)
    ]

    # save sanity check: predictions over training set
    sanity_mat = pd.DataFrame({
        country: regressions[i].predict(pop_data.loc[chosen_countries[i]].values.transpose())
        for i, country in enumerate(countries)
    }, index=pop_data.T.index).T
    sanity_mat.index.name = 'Country Name'
    sanity_mat.to_csv(submission_loc + '3population_sanity.csv')

    # save predictions
    pred_mat = pd.DataFrame({
        country: predictions[i]
        for i, country in enumerate(countries)
    }, index=pop_test.T.index).T
    pred_mat.index.name = 'Country Name'
    pred_mat.to_csv(submission_loc + '3population_predicted.csv')

    # save coefficient matrix
    coef_mat = pd.DataFrame({
        country: to_sparse_vector(list(chosen_countries[i].values), regressions[i].coef_, countries)
        for i, country in enumerate(countries)
    }, index=countries)
    coef_mat.index.name = 'Country Name'
    coef_mat.to_csv(submission_loc + '3parameters.csv')

    print(np.average(mse))