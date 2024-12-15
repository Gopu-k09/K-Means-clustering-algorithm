#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Input : Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# 2. Visualize the data
plt.scatter(X[:, 0], X[:, 1], s=30, c='gray')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3. Applying KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# output : Plot the results
plt.scatter(X[:, 0], X[:, 1], s=30, c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()        
plt.show()
