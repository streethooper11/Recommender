# Source for clustering using BERT vectors:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://techblog.assignar.com/how-to-use-bert-sentence-embedding-for-clustering-text/

import numpy as np
from sklearn.cluster import DBSCAN


def dbscanClustering(vectors, eps=0.5, min_samples=2, metric='euclidean'):
    """
    Clusters the BERT vectors using DBSCAN

    :param vectors: BERT vectors
    :param eps: Epsilon for DBSCAN
    :param min_samples: Minimum number of neighbours required
    :param metric: Distance measure
    :return: Result of DBSCAN clustering
    """

    # Change vectors to a numpy array
    x = np.array(vectors.tolist())
    print(x)

    # Perform DBSCAN on the numpy array and get labels
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm='auto').fit(x)
    DBSCAN_dataset = x.copy()
    DBSCAN_dataset.loc[:, 'Cluster'] = db.labels_

    return DBSCAN_dataset
