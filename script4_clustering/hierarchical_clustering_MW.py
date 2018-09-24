# K-Means Clustering

# Setup location of our datafile
datapath_root = "D:/Work/Udemy/Machine_Learning_Datasets/"
datapath_section = "Part 4 - Clustering/Section 24 - K-Means Clustering/"
datafile = "Mall_Customers.csv"

import os
os.chdir(datapath_root+datapath_section)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

# Importing the dataset
dataset = pd.read_csv(datafile)
#X = dataset.iloc[:, [3, 4]].values
X = dataset.iloc[:, [2,3,4]].values
# y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Make dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.show()

# Fitting hierarchical clustering 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')

# Fitting K-Means to the dataset
y_hc = hc.fit_predict(X)

# Visualising the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# using colorbrewer
import palettable.colorbrewer 
import matplotlib.colors as colors
cmap = colors.ListedColormap(palettable.colorbrewer.qualitative.Set1_6.mpl_colors)                      
# cmap = colors.ListedColormap(palettable.colorbrewer.diverging.Spectral_6.mpl_colors)                      


# Plot
for clusterIndex in range(6):
    ax.scatter(X[y_hc == clusterIndex, 0], X[y_hc == clusterIndex, 1], zs= X[y_hc == clusterIndex, 2], s = 100, c = cmap.colors[clusterIndex], label = 'Cluster ' + str(clusterIndex+1))
#ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], zs=kmeans.cluster_centers_[:, 2], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.legend()
plt.show()










