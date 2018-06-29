# -*- coding: utf-8 -*-
"""
iris_kmeans.py
Created on Mon Mar 13 01:42:37 2017

@author: Curtis
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load iris data set
iris = load_iris()

# Find a k-means clustering scheme
iris_clusters = KMeans(n_clusters=3, init="random").fit(iris.data)

# Visualize
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris_clusters.labels_)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()

n = 6