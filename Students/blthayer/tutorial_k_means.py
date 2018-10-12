"""Code for tutorial"""
# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

SEED = 4200

# Load data.
d = load_iris()
# Put into DataFrame
data = pd.DataFrame(d['data'], columns=d['feature_names'])

# Visualize.
fig, ax = plt.subplots()
f0 = d['feature_names'][0]
f1 = d['feature_names'][1]
ax.scatter(data[f0], data[f1], c=d['target'])
ax.set_xlabel(f0)
ax.set_ylabel(f1)
ax.set_title('Iris Data: {} vs {}'.format(f1, f0))

f2 = d['feature_names'][2]
f3 = d['feature_names'][3]
fig, ax = plt.subplots()
ax.scatter(data[f2], data[f3], c=d['target'])
ax.set_xlabel(f2)
ax.set_ylabel(f3)
ax.set_title('Iris Data: {} vs {}'.format(f2, f3))

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

# Visualize.
fig, ax = plt.subplots()
ax.scatter(data[f0], data[f1], c=labels)
# Plot mislabeled data.
ax.plot(data[f0].loc[mislabeled], data[f1].loc[mislabeled], color='r',
        marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')
# Plot centers.
ax.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='k',
        marker='+', markersize=16, markeredgewidth=3, linestyle='none')
ax.set_xlabel(f0)
ax.set_ylabel(f1)
ax.set_title('Clustered Data: {} vs {}'.format(f0, f1))

fig, ax = plt.subplots()
ax.scatter(data[f2], data[f3], c=labels)
# Plot mislabeled data.
ax.plot(data[f2].loc[mislabeled], data[f3].loc[mislabeled], color='r',
        marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')
# Plot centers.
ax.plot(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], color='k',
        marker='+', markersize=16, markeredgewidth=3, linestyle='none')
ax.set_xlabel(f2)
ax.set_ylabel(f3)
ax.set_title('Clustered Data: {} vs {}'.format(f2, f3))

plt.show()