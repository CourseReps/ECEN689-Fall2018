# Unsupervised Learning
By Prabhneet Arora and Sambandh Bhusan Dhal

## Contents
1. Need for Unsupervised Learning
2. Types of Unsupervised Learning with Demenstration of each type
3. Challenges 

## 1. Need For Unsupervised Learning (Explained with Real life examples)
a) Cancer Research: To find Gene Expression Levels in 100 patients with breast cancer by looking for sub-groups among the breast cancer 
samples to obtain a better understanding of the disease 

b) Online Shopping: To identify shopping behaviors and preference of shopping items by looking at the purchase history of customers

## 2. Types of Unsupervised Learning
### a) Clustering:

### (I) K- Means Clustering:
We need to prescribe the number of clusters we want to group the data into.
The clusters are non-overlapping i.e. no observation belongs to more than one cluster.
The main objective is to minimize "within-cluster" variation as much as possible.

#### Operating principle:
For each of the K clusters, we need to compute the cluster centroid.
We assign the point to the cluster whose centroid is the closest defined by the least Eucledian distance.


#### Demonstration:
![km1](https://user-images.githubusercontent.com/31993201/48086706-1c08a100-e1c3-11e8-9c1b-dc7cbe1153ce.png)
![km2](https://user-images.githubusercontent.com/31993201/48086705-1b700a80-e1c3-11e8-85b7-21a0b9776a03.png)


### (II) Hierarchial Clustering:
Here, we do not prescribe the number of clusters we group the data into.
It goes on to formulate a "tree-based" representation of the points called a "dendrogram" 
At the bottom, each point is a distinct leaf and as and when we move up the tree, similar points begin to fuse. The height of fusing/merging goes on to decide how similar the points are.

#### Operating Principle:
We need to start with each point as a separate cluster ( n clusters)
We calculate the "measure of dissimality" between the points and fuse the similar points and continue intil we get one cluster.
The "measure of dissimality" can be broadly classified into 2 types:-
     (a). On the basis of Eucledian distance:
         (i). Complete Linkage : Largest distance between observations
         (ii). Single Linkage : Smallest distance between observations
        (iii). Average Linkage : Average distance between observations
         (iv). Centroid Linkage : Distance between centroids of obseervations
     (b). On the basis of correlation

The different methods of calculating "measure of dissimalirity" would yield different results.

#### Demonstration:
![hc2](https://user-images.githubusercontent.com/31993201/48086703-1b700a80-e1c3-11e8-8521-43233be98b95.png)
![hc1](https://user-images.githubusercontent.com/31993201/48086704-1b700a80-e1c3-11e8-9b9e-f0fa7c698b5f.png)


## b) Association:
Association rules analysis is a technique to uncover how items are associated to each other.
There are 3 common ways to measure association:
1. Support
2. Confidence
3. Lift
e.g. Apriori Algorithm

#### Operating Principle of Apriori Algorithm:
We identify a particular characteristic of a dataset and attempt to note how frequently that characteristic pops up throughout the dataset.
A "frequent" dataset can  be characteristic is one that occurs above the pre-arranged amount, known as support.
Pruning helps to further differentiate between the categories that do and do not reach the overall support amount.
Next, the dataset is analyzed by looking for triplets. The triplets how even greater frequency. Analysis can detect more and more relations throughout the body of data until the algorithm has exhausted all of the possible.

#### Demonstration:
![pela](https://user-images.githubusercontent.com/31993201/48087214-62123480-e1c4-11e8-8749-39e886a0f6f8.png)

## c) Dimensionality Reduction:
It is used to find a low-dimensional representation of the observations that explain a good proportion of the variance.
The principal components are orthogonal ( uncorrelated).
They are ordered in decreasing order of the variance they capture: Z1 captures the highest variance, Z2 captures the second highest
variance and so on.
The principal components Z1,Z2.....Zq can be used in further supervised learning ( predictors in regression analysis).
e.g. Principal Component Analysis ( PCA).

#### Operating Principle of PCA:
The individual columns in X matrix are standardized ( mean zero and Standard Deviation one) before PCA is performed
The original variables in the X matrix are transformed so that the sum of squares of individual terms is maximized

#### Demonstration:
![untitled](https://user-images.githubusercontent.com/31993201/48085702-a4397700-e1c0-11e8-865d-350cff5dd64b.png)
![11](https://user-images.githubusercontent.com/31993201/48086022-6b4dd200-e1c1-11e8-88ac-340d7c125a0a.png)

## 3. Challenges of Unsupervised Learning:
* Tends to be more subjective
* No simple goal like prediction of a response
* Hard to access the results obtained and no way to check our work since we do not know the real "answer"

## References:

1. "An introduction to Statistical Learning with applications in R" by Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani
2.  https://www.youtube.com/watch?v=WGlMlS_Yydk
3.  https://www3.cs.stonybrook.edu/~cse634/lecture_notes/07apriori.pdf
4.  https://www.youtube.com/watch?v=T1ObCUpjq3o

