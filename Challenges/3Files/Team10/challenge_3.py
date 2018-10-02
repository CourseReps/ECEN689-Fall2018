
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from timeit import default_timer as timer
from sklearn.metrics import mean_squared_error as mse


def lasso_pred(x, y):
    
    lasso = Lasso(normalize = "True", tol = 0.00001, max_iter = 500000)
    w, q = x.shape
    
    coefs = []
    preds = []
    alpha_min=0
    alpha_max=0
    i = 0
    
    while i < q:
        
            start = timer() 
            print("\n------------------------------------\n")
            name ="Fitting for country no.: %s" %(i + 1)
            print(name)
      
        
            alpha_min = 1
            alpha_max = 10e15
            alpha_avg=(alpha_min + alpha_max)/2
            lasso.set_params(alpha = alpha_avg)
            population_vector = x.iloc[:, i]
            x.iloc[:, i] = np.zeros(w)
            y_true = y.iloc[:, i]
            lasso.fit(x, population_vector) 
            prediction = lasso.predict(y)
            error = mse(y_true, prediction)  
            x.iloc[:, i] = population_vector
            
            for j in range(150):
            
                #if(i == 60):
                    #print(alpha_min, alpha_max, np.count_nonzero(lasso.coef_))
                
                if np.count_nonzero(lasso.coef_) > 5 or np.count_nonzero(lasso.coef_) < 5:
                    
                    if np.count_nonzero(lasso.coef_) > 5:
                        
                        alpha_min = alpha_avg
                        alpha_avg=(alpha_min + alpha_max)/2
                        lasso.set_params(alpha = alpha_avg)
                        population_vector = x.iloc[:, i]
                        x.iloc[:, i] = np.zeros(w)
                        y_true = y.iloc[:, i]
                        lasso.fit(x, population_vector)
                        prediction = lasso.predict(y)
                        mean_error = mse(y_true, prediction)
                        x.iloc[:, i] = population_vector
                        
                    if np.count_nonzero(lasso.coef_) < 5:
                        
                        alpha_max = alpha_avg
                        alpha_avg=(alpha_min + alpha_max)/2
                        lasso.set_params(alpha = alpha_avg)
                        population_vector = x.iloc[:, i]
                        x.iloc[:, i] = np.zeros(w)
                        y_true = y.iloc[:, i]
                        lasso.fit(x, population_vector)
                        prediction = lasso.predict(y)
                        mean_error = mse(y_true, prediction)
                        x.iloc[:, i] = population_vector
                        name ="Fitting for country no.: %s" %(i + 1)
                
                else:
                    break
                
            assert np.count_nonzero(lasso.coef_) <= 5, "too,amu non-zero"
                
            if (mean_error < error):
                coefs_best = lasso.coef_
                pred_best = prediction
                error = mean_error
                alpha_value = alpha_avg
                country_number = np.count_nonzero(lasso.coef_)
     
            coefs.append(coefs_best)
            preds.append(pred_best)   
            
            end_timer = timer() - start
        
            time_statement = "Fitting time for country no. %s: " %(i + 1)
            print(time_statement + str(round(end_timer, 3)) + " seconds.")
            print("Alpha value: " + str(alpha_value))
            print("Number of countries used for fitting: " + str(country_number))
               
            i = i + 1
        
    return coefs, preds


def main():
    
    population_training_df = pd.read_csv('population_training.csv', encoding='cp1252')
    population_testing_df = pd.read_csv('population_testing.csv', encoding='cp1252')
    kaggle_file = pd.read_csv('population_sample_kaggle.csv', encoding='cp1252')
    parameter_file = pd.read_csv('population_parameters.csv', encoding='cp1252')
    
    population_training_df.drop(['Country Name'], axis=1, inplace=True)
    population_testing_df.drop(['Country Name'], axis=1, inplace=True)
    
    
    X = population_training_df.T
    Y = population_testing_df.T


    start = timer() 
    
    coefs, preds = lasso_pred(X, Y)
    
    preds = np.transpose(preds)
    coefs = np.transpose(coefs)
    
    print("\n\n")
    print(X.shape)
    print(Y.shape)
    print(np.shape(preds))
    print(np.shape(coefs))
    
    preds_1 = pd.DataFrame(data = preds)
    coefs_1 = pd.DataFrame(data = coefs)
 
    preds_1.reindex(index = kaggle_file.index, columns = kaggle_file.columns)
    coefs_1.reindex(index = parameter_file.index, columns = parameter_file.columns)
    preds_1.to_csv("predictions.csv")
    coefs_1.to_csv("coefficients.csv")
 
    timer_end = timer() - start
    
    print("\n\nTotal elapsed time: " + str(timer_end) + " seconds.")


if __name__ == "__main__":
    main()
  
