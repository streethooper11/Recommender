#!/usr/bin/env python3
"""
File responsible for clustering
"""
from matplotlib import pyplot as plt
# Source for clustering using BERT vectors:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://techblog.assignar.com/how-to-use-bert-sentence-embedding-for-clustering-text/

from sklearn.cluster import DBSCAN, KMeans


def dbscanClustering(vectors, eps, min_samples, metric='euclidean'):
    """
    Clusters the BERT vectors using DBSCAN

    :param vectors: List of vectors in numpy array
    :param eps: Epsilon for DBSCAN
    :param min_samples: Minimum number of neighbours required
    :param metric: Distance measure
    :return: Result of DBSCAN clustering
    """

    # Perform DBSCAN on the numpy array and get labels
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm='auto').fit(vectors)

    return db.labels_

def kmeansClustering(vectors, n_clusters):
    """
    Clusters the BERT vectors using K-Means Clustering

    :param vectors: List of vectors in numpy array
    :param n_clusters: Number of clusters
    :return: Result of K-Means clustering
    """

    # Perform K-Means Clustering on the numpy array and get labels
    db = KMeans(n_clusters=n_clusters, n_init="auto").fit(vectors)

    return db.labels_
