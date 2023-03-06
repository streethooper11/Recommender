# Source for clustering using BERT vectors:
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://techblog.assignar.com/how-to-use-bert-sentence-embedding-for-clustering-text/

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def dbscanClustering(vectors):
    """
    Clusters the BERT vectors using DBSCAN

    :param vectors: BERT vectors
    :return: Result of DBSCAN clustering
    """
    # Change vectors to a numpy array
    x = np.array(vectors)

    # Perform DBSCAN on the numpy array and get labels
    db = DBSCAN(eps=0.5, min_samples=2, metric='euclidean', algorithm='auto').fit(x)
    labels = db.labels_
