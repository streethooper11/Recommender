#!/usr/bin/env python3
"""
Responsible for generating actor information
"""

import numpy as np


def eliminateStopWords(actors, subwords, tensors, stopWordsLoc):
    # Read all stopwords by splitting them with whitespaces
    stopwords = set(open(stopWordsLoc).read().split())

    # Take all tensors as long as the matching subwords is not a stopword
    result = [t for a, s, t in zip(actors, subwords, tensors)
              if s not in stopwords]

    return result


def tensorsToNumpy(actors, subwords, tensors, save_loc, stopWordsLoc):
#    filtered_tensors = eliminateStopWords(subwords, tensors)

    # Change vectors to a numpy array
    x = []
    for eachTensor in tensors:
        x.append(eachTensor.tolist())
    x = np.array(x)

    np.save(save_loc, x)

    return None


def createDictionary_ClustersAndActors(clusters, actors):
    """
    Creates a dictionary from a list of clusters and a list of actors

    :param actors: List of actors
    :param clusters: List of clusters
    :return: A dictionary that consists of clusters as keys and a dictionary of actors and count as values
    """

    result = dict()
    i = 0

    while i < range(len(actors)):
        # make sure a cluster is assigned
        if clusters[i] != -1:
            # create a new list if this is the first time the cluster appears
            if clusters[i] not in result:
                result[clusters[i]] = dict()

            # create a new key-value pair if this is the first time the actor appears in the cluster with count as 0
            if actors[i] not in result[clusters[i]]:
                result[clusters[i]][actors[i]] = 0

            # increase count by 1
            result[clusters[i]][actors[i]] += 1

    return result, i
