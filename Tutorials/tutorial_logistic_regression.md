# Logistic Regression
by Jatin Kamnani and Venkata Pydimarri

Problem Description 
The problem in the tutorial we dealt with is to decide whether a student gets an admit in a university after application. Hence, this problem can be considered to be a binary classification problem.

Understanding the differences between Supervised Learning and Unsupervised Learning

Supervised Learning :- It is the task of learning a function that maps an input to an output based on example input-output pairs.Regression and Classification problems are examples of Supervised Learning.

Unsupervised Learning :- It is the task of finding structure within data without having knowledge of input-output pairs. Clustering & problem is the well known example for unsupervised learning.

Classification Problem :- It deals with identifying the class to which a given data belongs. A prior knowledge of examples of data belonging to each class has to be known. Classification problem can be either a binary classification or a multi-class classification problem.

What is Logistic Regression?

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.

Features of Logistic Regression

1. In logistic regression, the output or target variable is binary.
2. The hypothesis function always stays bounded between 0 &1. [ 0<=hθ(x)<=1]
3. Hypothesis Representation for Logistic Regression :
          1. hθ(x)= ø(θTx) with z= ΘTx
          2. ø(z) = 1/(1+ e^-z)

hθ(x) = Estimated probability that y=1 on input x. Output y is predicted as 1 when hθ(x)>=0.5 and 0 otherwise.

Features considered for the implementation :-

1.GRE Score
2.TOEFL Score
3.GPA
4.Number of publications
5.Number of projects

Implementation Procedure :-

1. Main function is created and admit statistics are loaded into pandas dataframe. 
2. The dataset is split into train and test dataset
3. Understanding the dataset:
  The features of the data are noted and taken as input to the program. The outcomes of a particular feature for a given category of target variable is stored in the variables representing the output categories. 

4. The training dataset to model the logistic regression
5. The accuracy of model on training dataset is calculated.
6. Similarly, the test dataset accuracy is computed.
