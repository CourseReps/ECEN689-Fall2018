import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

population_training_df = pd.read_csv('population_training_kaggle.csv', encoding='cp1252')
population_testing_df  = pd.read_csv('population_testing_kaggle.csv', encoding='cp1252')

print(population_training_df.shape)
print(population_testing_df.shape)

pop_train_data = population_training_df
pop_test_data = population_testing_df

pop_train_data.drop(['Id'], axis=1,inplace=True)

year_col_bkp = pop_test_data['Id']
pop_test_data.drop(['Id'], axis=1, inplace=True)

alphas = 10**np.linspace(5,-1,1000)*1
alphas

coefpara_array = []
pred_pop_array = []

lassomodel = Lasso(fit_intercept=True, normalize=True)

for country in range(0, 258):

    min_error = float('inf')
    coef_min_err = 0

    test_x = pop_test_data.drop(pop_test_data.columns[[country]], axis=1, inplace=False)
    test_y = pop_test_data[pop_test_data.columns[country]]

    train_x = pop_train_data.drop(pop_train_data.columns[[country]], axis=1, inplace=False)
    train_y = pop_train_data[pop_train_data.columns[country]]

    for x in alphas:

        lassomodel.set_params(alpha=x)
        lassomodel.fit(train_x, train_y)
        pred_pop = lassomodel.predict(test_x)
        error = mean_squared_error(pred_pop, test_y)
        count_non_zero = np.count_nonzero(lassomodel.coef_)
        if count_non_zero <= 5 and count_non_zero > 0:
            if error < min_error:
                coef_min_err = lassomodel.coef_
                min_error = error
                pred_pop = lassomodel.predict(test_x)
                arr_coef_min_err = coef_min_err
                final_coef_arr = np.insert(arr_coef_min_err, country, 0)
    coefpara_array.append(final_coef_arr)
    pred_pop_array.append(pred_pop)

final_coef = np.array(coefpara_array)

countries = population_training_df.columns

coefficient_df = pd.DataFrame(data=coefpara_array, columns=countries, index=countries)
final_coefficient_df = coefficient_df.transpose()
final_coefficient_df.to_csv('population_parameters.csv', encoding='cp1252')

pred_array = np.array(pred_pop_array)

actual_df = pd.DataFrame(data=pred_pop_array)
transpose_df = actual_df.transpose()
transpose_df_final = pd.DataFrame(year_col_bkp, columns=['Id'])


for x in range(len(population_testing_df.columns)):
    cols = population_testing_df.columns[x]
    transpose_df_final[cols] = transpose_df[x]


transpose_df_final.to_csv('population_prediction.csv',encoding='cp1252', index=False)
