
# Introduction to L1 and L2 regularization

The motivation behind pursuing regularisation is to avoid the tendency of the model to capture noise while learning patterns in data.
We want to create models that generalise and perform well on different data-points - AVOID Overfitting!

In this tutorial, We have explained the complex science behind ‘Ridge Regression‘ and ‘Lasso Regression‘ which are the most fundamental regularization techniques.

What to do when p >> n ?
Collect more data or reduce the existing number of features
There are situations where we cannot afford collecting more data, so the only option we have is to decrease 'p' somehow i.e the model should only consider useful features


# Ridge Regression

It performs ‘L2 regularization’, i.e. adds penalty equivalent to square of the magnitude of coefficients. Thus, it optimises the following:

Objective = RSS + α * (sum of square of coefficients)
Here, α(alpha) is the tuning parameter which balances the amount of emphasis given to minimising RSS vs minimising sum of square of coefficients

# Lasso Regression

LASSO stands for Least Absolute Shrinkage and Selection Operator. I know it doesn’t give much of an idea but there are 2 key words here - absolute and selection.

Lasso regression performs L1 regularization, i.e. it adds a factor of sum of absolute value of coefficients in the optimisation objective.

Objective = RSS + α * (sum of absolute value of coefficients)

# Key Notes

Need to Standardise Features
While performing regularization techniques, you should standardise your input dataset so that it is distributed according to N(0,1), since solutions to the regularised objective function depend on the scale of your features.

A technique known as Elastic Nets, which is a combination of Lasso and Ridge regression is used to tackle the limitations of both Ridge and Lasso Regression

