#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:53:54 2018

Name: khalednakhleh

This file includes task 1, task 3, and task 4.

Task 2 (decision tree) is in the other Python script file.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from timeit import default_timer as timer

"""----------------------------------------------------------------------------------------- """


def lin_reg_white(white_train, white_test):
    """ Create a linear regression model for the white wine data set.
    Predicting the white wine quality and comparing it with the test one.
    Using the same white wine model on the white wine dataset.
    """
    
    # Slicing the white wine training dataset into parameters and quality 
    x = white_train.iloc[:, 1: 12]
    y = white_train.iloc[:, 12]
    # Selecting the test data to predict white wine quality
    t = white_test.iloc[:, 1: 12]
    
    start = timer()                            # Start timer
    pred = lr()                                # Assigning the model name
    pred.fit(x,y)                              # Predictor training
    pred.end = timer() - start                 # End timer
    pred.quality = pred.predict(t)             # Predicting correct labels
    
    # Printing runtime for this model
    print("\nLogistic Regression run time for white wine: \n\n", \
          round(pred.end, 5), " seconds.\n\n\n")
    
    return pred


def lin_reg_red(red_train, red_test):
    """ Create a linear regression model for the red wine data set.
    Predicting the red wine quality and comparing it with the test one.
    Using the same red wine model on the red wine dataset.
    """
    
    # Slicing the white wine training dataset into parameters and quality 
    x = red_train.iloc[:, 1: 12]
    y = red_train.iloc[:, 12]
    # Selecting the test data to predict white wine quality
    t = red_test.iloc[:, 1: 12]
    
    start = timer()                            # Start timer
    pred = lr()                                # Assigning the model name
    pred.fit(x,y)                              # Predictor training
    pred.end = timer() - start                 # End timer
    pred.quality = pred.predict(t)             # Predicting correct labels
    
    # Printing runtime for this model
    print("\nLogistic Regression run time for red wine: \n\n", \
          round(pred.end, 5), " seconds.\n\n\n")
    
    return pred

def white_red(white_train, red_test):
    """ Create a linear regression model from the white wine data set.
    Predicting the red wine quality using model fit for white wine.
    Using this model created from white wine on red wine.
    """
    
    # Slicing the white wine training dataset into parameters and quality 
    x = white_train.iloc[:, 1: 12]
    y = white_train.iloc[:, 12]
    # Selecting the test data to predict red wine quality
    t = red_test.iloc[:, 1: 12]
    
    start = timer()                            # Start timer
    pred = lr()                                # Assigning the model name
    pred.fit(x,y)                              # Predictor training
    pred.end = timer() - start                 # End timer
    pred.quality = pred.predict(t)             # Predicting correct labels
    
    # Printing runtime for this model
    print("\nLogistic Regression run time for red wine using white wine model: \n\n", \
          round(pred.end, 5), " seconds.\n\n\n")
    
    return pred

def main():
  
    # Importing files using Pandas
    # Red wine files
    red_train = pd.read_csv("winequality-red-training.csv")
    red_test = pd.read_csv("winequality-red-testing.csv")
    red_output = pd.read_csv("winequality-red-sample.csv")
    
    # White wine files
    white_train = pd.read_csv("winequality-white-training.csv")
    white_test = pd.read_csv("winequality-white-testing.csv")
    white_output = pd.read_csv("winequality-white-sample.csv")

# TASK 1       
###########################################

 
    # Linear regression model for white wine
    predictor_white = lin_reg_white(white_train, white_test) 
    white_output.iloc[:, 1] = predictor_white.quality
    
    param = pd.DataFrame(data = predictor_white.coef_)
    
    
    # Outputting results to winequality-white-sample.csv  
    white_output.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-white-sample.csv", index = False)
    
    white_output.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-white-solution.csv", index = False) 
     
    param.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-white-parameters.csv", index_label = "Id") 
  
  # TASK 3      
###########################################
 
    # Linear regression model for red wine using red wine training data
    predictor_red = lin_reg_red(red_train, red_test) 
    red_output.iloc[:, 1] = predictor_red.quality
    
    # Outputting results to winequality-red-solution.csv  
    red_output.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-red-solution.csv", index = False)
    
    # Outputting results to winequality-red-sample.csv  
    red_output.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-red-sample.csv", index = False)    
        
 # TASK 4      
###########################################

 
    # Linear regression model for red wine using white wine training data
    white_2 = white_red(white_train, red_test) 
    red_output.iloc[:, 1] = white_2.quality
    
    # Outputting results to winequality-red-solution-white-model.csv  
    red_output.to_csv("/Users/khalednakhleh/Documents/ecen_689" \
                  "/challenge_4/winequality-red-solution-white-model.csv", index = False)
    
 

# Initializing the file
if __name__ == "__main__":
    main()
    
    

    
    
    

    
    
    
    
    