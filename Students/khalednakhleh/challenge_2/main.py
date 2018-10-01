#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:21:29 2018

Name: khalednakhleh
"""

import pandas as pd
import de_idx as dx
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knn
from timeit import default_timer as timer

"""----------------------------------------------------------------------------------------- """

def log_reg(x, y, t, q):
    """ This function is an amalgamation of different minute tasks that 
    I just gatherd into a singal call function to ease work."""
    
    start = timer()                            # Start timer
    pred = lr(solver = "saga", tol = 0.01, max_iter = 600, multi_class = "multinomial")
    pred.fit(x,y)                              # Predictor training
    pred.error = 1 - pred.score(t,q)           # Predictor test
    pred.end = timer() - start                 # End timer
    pred.labels = pred.predict(t)              # Predicting correct labels
    
    # Printing some information for user
    print("------------------------------------------")
    print("\nLogistic Regression error percentage: " + str(round((pred.error * 100), 5)) + " %")
    print("\nLogistic Regression run time: ", round(pred.end, 5), " seconds.\n\n\n")
    
    return pred

def k_nn(x, y, t, q, n):
    """K-nearest neighbors algorithm. K is recommended to be an odd number,
    and not a multiple of the number of classes. """
    
    start = timer()                            # Start timer
    k_pred = knn(n_neighbors = n, n_jobs = -1) # set predictor
    k_pred.fit(x, y)                           # Predictor training
    k_pred.error = 1 - k_pred.score(t, q)      # Predictor test
    k_pred.labels = k_pred.predict(t)          # Predicting correct labels
    k_pred.end = timer() - start               # End timer
  
    # Printing some information for user
    print("------------------------------------------")     
    print("\nK-nearest neighbors error percentage: " + str(round((k_pred.error * 100), 5)) + " %")
    print("\nK-nearest neighbors run time: ", round(k_pred.end, 5), " seconds.\n\n\n")
    
    return k_pred

def main():
    """ Files are inserted, and apply log_reg and knn functions on the files.
    the function also creates three .csv files for weight vectors,
    log_reg results, and knn results."""
    
    # user inputs file names in .csv format
    training_file = pd.read_csv("mnist_train.csv", index_col = 0)
    test_data = pd.read_csv("mnist_test.csv", index_col = 0)
    
    # Taking all data values from .csv file
    train_data = training_file.iloc[:, 1:]
    # Taking all train labels from the "Category" column 
    train_label = training_file.iloc[:, 0]
    # Taking test labels from idx file
    test_label = (dx.De_idx_label("t10k-labels-idx1-ubyte")).array
    # K value for K-NN algorithm
    k = 103

    # Running the two algorithms for given input values
    predictor = log_reg(train_data, train_label, test_data, test_label)
    k_class = k_nn(train_data, train_label, test_data, test_label, k)
    
    # Creating Pandas dataframes for three output .csv files
    weight = pd.DataFrame(data = predictor.coef_ , columns = train_data.columns)
    lr_values = pd.DataFrame(data = predictor.labels, columns = ["Category"])
    k_class_values = pd.DataFrame(data = k_class.labels, columns = ["Category"])
    
    weight.index += 1
    lr_values.index += 1
    k_class_values.index += 1
    
    # Output values to three .csv files
    weight.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_2/2challenge_logreg_vectors.csv", index_label = "Id")
    
    lr_values.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_2/2challenge_logreg.csv", index_label = "Id")
    
    k_class_values.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_2/2challenge_knn.csv", index_label = "Id")
    
    
if __name__ == "__main__":
    
    main()

