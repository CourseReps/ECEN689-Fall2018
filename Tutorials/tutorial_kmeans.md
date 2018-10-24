# K-Means Clustering
by Brandon Thayer and Harish Chigurupati

## Contents
* Introduction to Clustering
* K-Means Clustering- Basic Understanding
* K-Means Mathematical Representation
* K-Means Algorithm
* Demonstration
* Pros and Cons
* References


## Introduction to Clustering
Clustering is one of the most widely used techniques for exploratory data analysis. Clustering is the task of grouping a set of objects such that similar objects end up in the same group and dissimilar objects are separated into different groups. In simple words, the aim is to segregate groups with similar traits and assign them into clusters. 

There are various clustering algorithms that are available. We choose the right algorithm based on the data we are handling. .I.e, we identify meaningful groups with a lack of "ground truth" for clustering, which is a common problem in unsupervised learning.  The goal of supervised learning is clear we wish to learn a classifier which will predict the labels of future examples as accurately as possible. Furthermore, a supervised learner can estimate the success, or the risk, of its hypotheses using the labeled training data by computing the empirical loss. In contrast, clustering is an unsupervised learning problem; namely, there are no labels that we try to predict. Instead, we wish to organize the data in some meaningful way. As a result, there is no clear success evaluation procedure for clustering. In fact, even on the basis of full knowledge of the underlying data distribution, it is not clear what is the correct" clustering for that data or how to evaluate a proposed clustering.

<p align="center">
<img src="https://github.com/CourseReps/ECEN689-Fall2018/blob/master/Students/harishchigurupati/Images%20for%20tutorial/clusters.png" width="600" height="400">
</p>


## K-Means Clustering- Basic Understanding

K-means clustering is a simple Clustering algorithm coming under unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). Here, the user specifies the number of clusters the data is to be partitioned (denoted by "K"). The main aim of this algorithm is to minimise the "inertia"(within cluster sum of squares). Since it is a very simple yet efficient algorithm, K-Means is very fast as it converges to local minima rapidly.

<p align="center">
<img src="https://github.com/CourseReps/ECEN689-Fall2018/blob/master/Students/harishchigurupati/Images%20for%20tutorial/k_means_cluster.png" width="300" height="400">
</p>

## K-Means Mathematical Representation
* In k-means the data X is partitioned into disjoint sets C1,...,Ck where each Cj is represented by a centroid μ.
*  The k-means objective function measures the squared distance between each point in X to the centroid of its cluster.
* Each input sample belongs to the closest Cj.
* Mathematically, K-Means Objective function is given as:

![first equation](http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%20%5Cunderset%7B%5Cmu_j%20%5Cin%20C%7D%7B%5Ctext%7Bmin%7D%7D%28%7C%7Cx_i%20-%20%5Cmu_j%7C%7C%5E2%29)

## K-Means Algorithm
* Finding the optimal k-means solution is often computationally infeasible.
* To circumvent this, simple iterative algorithm used.
* This Iterative algorithm converges to local minimum.
 **Algorithm**
1. *Initialize*: Randomly choose initial centroid μi,...,μk.
2. *Repeat until convergence:*

2.1  Assign data to clusters:

![Second Equation](http://latex.codecogs.com/gif.latex?%24%5Cforall%20i%20%5Cin%20%5Bk%5D%24%20set%20%24C_i%20%3D%20%5C%7Bx%20%5Cin%20X%3A%20i%20%3D%20%5Ctext%7Bargmin%7D_j%20%7C%7Cx%20-%20%7B%5Cmu%7D_j%7C%7C%5C%7D%24)

2.2 *Update centroids:*

![third equation](http://latex.codecogs.com/gif.latex?%5Cforall%20i%20%5Cin%20%5Bk%5D%20update%20%7B%5Cmu%7D_i%20%3D%20%5Cfrac%7B1%7D%7B%7CC_i%7C%7D%5Csum_%7Bx%20%5Cin%20C_i%7D%20x)

## Demonstration
```python
"""Code for tutorial"""
# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

SEED = 4200

# Load data.
d = load_iris()
# Put into DataFrame
data = pd.DataFrame(d['data'], columns=d['feature_names'])

# Grab feature names for easy access
f0 = d['feature_names'][0]
f1 = d['feature_names'][1]
f2 = d['feature_names'][2]
f3 = d['feature_names'][3]

# Define basic color map
colors = ['m', 'y', 'c']


# Simple function for scatter plots. Hard-coded.
def plot_scatter(axis, x, y, targets, target_labels, colors, xlabel, ylabel):
    for k in range(3):
        # Boolean vector
        b = targets == k
        axis.scatter(x=x[b], y=y[b], c=colors[k], label=target_labels[k])

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend()

    return axis


# Visualize.
fig, ax = plt.subplots()
ax = plot_scatter(axis=ax, x=data[f0], y=data[f1], targets=d['target'],
                  target_labels=d['target_names'], colors=colors, xlabel=f0,
                  ylabel=f1)
ax.set_title('Iris Data: {} vs {}'.format(f1, f0))

fig2, ax2 = plt.subplots()
ax2 = plot_scatter(axis=ax2, x=data[f2], y=data[f3], targets=d['target'],
                   target_labels=d['target_names'], colors=colors, xlabel=f2,
                   ylabel=f3)
ax2.set_title('Iris Data: {} vs {}'.format(f2, f3))

# Cluster.
km = KMeans(n_clusters=3, random_state=SEED)
km.fit_predict(data)

# Map the labels so our colors match (hard-coding).
labels = km.labels_
zero_labels = labels == 0
one_labels = labels == 1
# two_labels = labels == 2
labels[zero_labels] = 1
labels[one_labels] = 0
# labels[two_labels] = 2

# Get labels we mis-classified.
mislabeled = (d['target'] - labels) != 0

# Compute percent accuracy:
acc = (1 - np.count_nonzero(mislabeled) / mislabeled.shape[0]) * 100
print('K-Means correctly identified {:.0f}% of the data.'.format(acc))

# Visualize.
fig3, ax3 = plt.subplots()
ax3 = plot_scatter(axis=ax3, x=data[f0], y=data[f1], targets=labels,
                   target_labels=d['target_names'], colors=colors, xlabel=f0,
                   ylabel=f1)
ax3.set_title('Clustered Data: {} vs {}'.format(f0, f1))

fig4, ax4 = plt.subplots()
ax4 = plot_scatter(axis=ax4, x=data[f2], y=data[f3], targets=labels,
                   target_labels=d['target_names'], colors=colors, xlabel=f2,
                   ylabel=f3)
ax3.set_title('Clustered Data: {} vs {}'.format(f2, f3))

# Plot mislabeled data.
ax3.plot(data[f0].loc[mislabeled], data[f1].loc[mislabeled], color='r',
         marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')
ax4.plot(data[f2].loc[mislabeled], data[f3].loc[mislabeled], color='r',
         marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')

# Plot centers.
ax3.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b',
         marker='+', markersize=16, markeredgewidth=3, linestyle='none')
ax4.plot(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], color='b',
         marker='+', markersize=16, markeredgewidth=3, linestyle='none')

# Perform K-Means with normalized data.
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
km2 = KMeans(n_clusters=3, random_state=SEED)
km2.fit_predict(data_scaled)
# Map the labels so our colors match (hard-coding).
labels2 = km2.labels_
zero_labels = labels2 == 0
one_labels = labels2 == 1
two_labels = labels2 == 2
labels2[zero_labels] = 2
labels2[one_labels] = 0
labels2[two_labels] = 1

# Get labels we mis-classified.
mislabeled2 = (d['target'] - labels2) != 0

# Compute percent accuracy:
acc2 = (1 - np.count_nonzero(mislabeled2) / mislabeled2.shape[0]) * 100
print(('K-Means with normalized data correctly identified {:.0f}% of the '
       + 'data.').format(acc2))

```
## Pros/Cons
**Pros**
* K-Means Algorithm is very easy to implement as the euclidian distance calculation knowledge is sufficient to perform this algorithm.
* It is very fast when compared to other Clustering algorithms.
* It is very simple to understand.
* This Algorithm always converges to a local minima. Hence it always produces a result.

**Cons**
* Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.
* Large feature space can inflate the Euclidean distance.
* User must select number of clusters-this means that the user must know the data in depth to guess or fix the correct value for "K".
* Converges to local minimum depending on starting point-Circumvent by repeating algorithm with different initial centroids.

## References
* [Wikipedia](https://en.wikipedia.org/wiki/Cluster_analysis)
* [scikit-learn](http://scikit-learn.org/stable/modules/clustering.html)
* [Course textbook](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)

