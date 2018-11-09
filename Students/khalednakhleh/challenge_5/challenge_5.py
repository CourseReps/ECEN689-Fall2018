#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:18:58 2018

Name: khalednakhleh
"""

from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plotting_decision_region(x, y, classifier, resolution = 0.01, title = "example"):
    """ Plotting decision regions for the passed classifer.
    -------------------------------------------------------
    x = set of training samples (n,m)
    y = set of labels
    classifier = training model being used
    resolution = best be kept at 0.01. anything more and computation time increases
    title = plot title
    """
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'green', 'lightgreen', 'gray', 'cyan')
    
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                     np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        
        plt.scatter(x = x[y == cl, 0], y = x[y == cl, 1],
                alpha = 0.8, c = cmap(idx),
                marker = markers[idx])
        plt.title(title)
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.savefig(title + ".png")  

def main():
    
    # Choose kernel from "linear", "sigmoid", "rbf", and "poly"
    kernel = "rbf"
    # pulling .csv files
    train = pd.read_csv("5challenge_training_khaled.jamal.csv", index_col = 0)
    test = pd.read_csv("5challenge_testing_khaled.jamal.csv", index_col = 0)
    
    # locating samples and labels    
    x = train.iloc[:, 1:].values
    y = train["Class"].values  
    p = test.iloc[:, 1:].values
    
    # calling, training, and predicting with the SVM model
    model = svm.SVC(gamma='auto_deprecated', kernel = kernel,
                    max_iter = -1, tol = 0.0001,
                    degree = 3, shrinking = True)
    model.fit(x, y)       
    prediction = model.predict(p)
    
    # saving predictions to the test file    
    test["Class"] = prediction   
    test.to_csv("5challenge_testing_khaled.jamal.csv")
    
    # plotting figures for training and testing data
    plt.figure(1)
    plotting_decision_region(x, y, model, title = "plotting training data using " + kernel + " kernel")
    
    plt.figure(2)
    plotting_decision_region(p, prediction, model, title = "plotting test data using " + kernel + " kernel")


if __name__ == "__main__":
    main()
