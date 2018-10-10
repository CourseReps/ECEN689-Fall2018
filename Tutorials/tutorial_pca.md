# Principal Component Analysis
### by Aditya Lahiri and Shirish Pandagare
### Instructor: Prof. Jean-Francois Chamberland

### Introduction 
PCA is an unsupervised dimensionality reduction technique that reduces the dimensions of a d-dimenisional dataset by projecting it onto a k-dimenisional subspace (k < d), while reatining as much of the variation of the data as possible. PCA is linear transformation that finds new sets of axes for the data. It does so by finding the direction of maximum variance and setting it as the first axis often known as principle component 1 and then it finds the direction of second most variance and set its as the second axis or principle component 2 and so on. For d-dimensional dataset it is only possible to find 'd' principle components and these princple components are orthogonal and uncorrelated to each other. 

### Why use PCA
Some of the applications of PCA are: 
Visualization
Feature extraction
Noise filtering 
Stock market prediction
Gene data analysis

### How to perform PCA on a dataset. 
1. Standardize the dataset.
2. Construct a covariance or correlation (for dataset containing highly varying features) matrix for the standardized dataset. 
3. Obtain the eigen values and eigen vectors from the the covariance or correlation matrix.
4. Sort the eigen values in descending order. 
5. To reduce the dataset from d to k(<d) dimensions, select k eigenvectors associated with the k largest eigen vectors.
6. Construct a projection matrix (W) from the k selected eigen vectors.
7. Transform the original dataset via the projection matrix W to obtain a new dataset with k-dimensions.
8. To find the proportion of variation explained by the new set of axes divide the sum of the k largest eigen values by sum of all the eigen values.

### Interactive tutorial for PCA
Please use the following link for an interactive tutorial on PCA
http://setosa.io/ev/principal-component-analysis/

### Python implementation of PCA.
We use the wine dataset in scikit learn library to demonstrate the applications of PCA. We perform classification on this dataset which contatins 3 class labels using logistic regression. We compare the accuracies of logistic regression in predicting the class labels by first training and testing it on the oridnary dataset and then training and testing on the PCA transformed dataset. We construct the confusion matrix for each case and find that PCA improves the accuracy for test prediction. We also see that pca allows us to visualize our test results by projecting our 13 dimensional data onto a 2 dimesional subspace. 


```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Load the wine data from scikit learn

from sklearn.datasets import load_wine
wine = load_wine()

#### Display the wine dataset Objective: Classify wine into 3 categories

df =  pd.DataFrame(np.c_[wine['data'], wine['target']],columns= np.append(wine['feature_names'], ['target']))
df.head(150) # 178 by 14

### Split the dataset into training (75%)and test set (25%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=100)

### Scale the training and testing dataset, in order to get proper projections of the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)# Scale train data
X_test = sc.transform(x_test)# scale test data

### Transform the training and test set to the new PCA axis, also specify the number of components
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) # use None # use 2
X_train_pca = pca.fit_transform(X_train) # PCA Transform training set with n_components features
X_test_pca = pca.transform(X_test)# PCA Transform the testing set
explained_variance = pca.explained_variance_ratio_
print(explained_variance)# Get the explained variance
print(np.sum(explained_variance))# Total variance explained
### Plot the variance explained by each of the PCs
cum_sum=pca.explained_variance_ratio_.cumsum()
fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(explained_variance.size), explained_variance, color = 'b',alpha=0.5)
plt.title("Plot of variance explained by PCs ")
### Classify using Logistic Regression
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(random_state = 0)
clf.fit(x_train,y_train)

classifier_pca = LogisticRegression(random_state = 0)
classifier_pca.fit(X_train_pca, y_train)

### Predict the test label

y_pred=clf.predict(X_test)
y_pred_pca = classifier_pca.predict(X_test_pca)

### Construct a confusion matrix for simple logistic regression classifier
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

### Construct a confusion matrix for logistic regression classifier with PCA transformed data
from sklearn.metrics import confusion_matrix
cm_pca = confusion_matrix(y_test, y_pred_pca)
print(cm_pca)

### Visual representation of test data classification on the two PC axi

from matplotlib.colors import ListedColormap
X_set, y_set = X_test_pca, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_pca.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
markers=['D','v','*']
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],s=100,marker=markers[j],edgecolors='k',
                c = ListedColormap(('red','green','blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
```
### Drawbacks of PCA
The the new axis formed by PCA are not interpretable.
PCA assumes that the input data is real and continuous.


### References
https://www.dezyre.com/data-science-in-python-tutorial/principal-component-analysis-tutorial
http://setosa.io/ev/principal-component-analysis/
http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf
