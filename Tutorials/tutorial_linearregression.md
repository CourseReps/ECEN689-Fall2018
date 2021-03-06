
# Linear Regression
### by Samyuktha Sankaran & Harinath PRS
### Instructor: Prof. Jean-Francois Chamberland

### Abstract 
We had presented a tutorial on Linear regression. We explained the concept and the mathematics behind the model and demonstrated the same with a sample code. We had also discussed some possible applications and anomalies in the model.

### Linear Regression

Linear Regression is a type of supervised model, used to find the relationship between the predictors and the target variables. There are two types of Linear regression: Simple and Multiple Linear regression. (SLR,MLR)
Simple Linear Regression is where a relationship is established between two quantities - one is the independent variable, that is the predictor and the dependent variable, the output. The idea is to obtain a line that best fits the data such a way that the total error is minimal.
\begin{equation}
    Y_p = \beta_0 + \beta_1X 
\end{equation}
\begin{equation}
error = \sum_{i=1}^{n} (actual\ value - predicted\ value)^2
\end{equation}
The error is squared, so that the positive and negative terms do not get cancelled. If β1 > 0 implies that a positive relation exists between the predictor and the target. And if β1 < 0, there is a negative relationship between the predictor and target.

### Metrics for model evaluation 
1. RSS The Residual Sum of squares gives information about how far the regression line is from the average of actual output.
2. Sum of squared error tells how much the target values vary around the regression line. 
3. Total Sum of squares is how much the data points are scattered about the mean.
4. P-Value describes the relation between the null hypothesis and predicted value. A high P value would mean that changes in the predictor have no effect on the target variable. A low P value rejects the null hypothesis indicating that there is a relation between the target and the predictor variables.

Multiple Linear regression attempts to model a relationship between two or more predictor variables and the target variable by fitting a linear equation to the observed data. Every value of the independent variables x are associated with the dependent variable y.
\begin{equation}
    Y_p = \beta_0 + \beta_1X_1 + \beta_2X_2 + .... + \beta_nX_n
    \end{equation}
An extension to the linear regression model is the polynomial regression, where a non linear
equation would be a best fit for the observed data set.
\begin{equation}
    Y_p = \beta_0 + \beta_1X_1 + \beta_2X_2^2
    \end{equation}

### Demonstration
We demonstrated the working of a linear regression model through a small code snippet. We have used the mpg datasets that estimates the miles per gallon for automobiles based on parameters like number of cylinders, displacement, horsepower, power, weight, acceleration and model manufactured year.

### Implementation

1. After loading the train and test datasets, process the datasets by dropping entries that have missing entries. One-hot encoding was applied to the origin of the automobile predictor variable.
2. Import the linear regression model from the SciKit learn package.
3. Fit the training data to the model and print the coefficients of the respective predictor variable.The coefficients comments about the effect the predictor has on the target variable.
4. Pass the test data set and obtain the accuracy and mean squared error.
5. Plot the obtained results.

