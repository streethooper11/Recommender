#!/usr/bin/env python3
"""
Responsible for creating and processing of data structures and saving numpy arrays to files.
"""
import pandas as pd


def tensorsToDF(actors: list, vectors, save_loc):
    # Change vectors to a pandas dataframe
    x = []
    for eachVector in vectors:
        x.append(eachVector.tolist())  # this converts the tensor to a regular list

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    d = {'actors': actors, 'vectors': x}
    df = pd.DataFrame(data=d)
    df.to_csv(save_loc)

    return df


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
