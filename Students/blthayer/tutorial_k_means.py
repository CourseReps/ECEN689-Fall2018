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

# Visualize.
fig, ax = plt.subplots()
ax.scatter(data[f0], data[f1], c=d['target'])
ax.set_xlabel(f0)
ax.set_ylabel(f1)
ax.set_title('Iris Data: {} vs {}'.format(f1, f0))

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

# Compute percent accuracy:
acc = (1 - np.count_nonzero(mislabeled) / mislabeled.shape[0]) * 100
print('K-Means correctly identified {:.0f}% of the data.'.format(acc))

# Visualize.
fig_1, ax_1 = plt.subplots()
ax_1.scatter(data[f0], data[f1], c=labels)
ax_1.set_xlabel(f0)
ax_1.set_ylabel(f1)
ax_1.set_title('Clustered Data: {} vs {}'.format(f0, f1))

fig_2, ax_2 = plt.subplots()
ax_2.scatter(data[f2], data[f3], c=labels)
ax_2.set_xlabel(f2)
ax_2.set_ylabel(f3)
ax_2.set_title('Clustered Data: {} vs {}'.format(f2, f3))

# Plot mislabeled data.
ax_1.plot(data[f0].loc[mislabeled], data[f1].loc[mislabeled], color='r',
          marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')
ax_2.plot(data[f2].loc[mislabeled], data[f3].loc[mislabeled], color='r',
          marker='x', markersize=8, markeredgewidth=1.5, linestyle='none')

# Plot centers.
ax_1.plot(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='b',
          marker='+', markersize=16, markeredgewidth=3, linestyle='none')
ax_2.plot(km.cluster_centers_[:, 2], km.cluster_centers_[:, 3], color='b',
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
