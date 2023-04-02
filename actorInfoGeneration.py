#!/usr/bin/env python3
"""
Responsible for generating actor information with cluster information,
"""

import pandas as pd

def createDictionary_ClustersActorsRatings(clusters, actors, ratingCsvLoc):
    """
    Creates a dictionary from a list of clusters and a list of actors

    :param clusters: List of clusters
    :param actors: List of actors to be used in clusters
    :param ratingCsvLoc: CSV file of movie ratings that each actor participated in
    :return: A dictionary with cluster information, a dictionary with total movie ratings, a dictionary with
            number of actor appearance with movie ratings
    """

    result_clusters = dict()
    result_ratings = dict()
    result_ratings_appearance = dict()

    # aggregate cluster counts
    for i in range(len(actors)):
        # Check if this is the first time the actor appears
        if actors[i] not in result_clusters:
            result_clusters[actors[i]] = dict()

        if clusters[i] != -1:
            # If this is the first time the cluster appears, initialize it as 0
            if clusters[i] not in result_clusters[actors[i]]:
                result_clusters[actors[i]][clusters[i]] = 0

            # increase count by 1
            result_clusters[actors[i]][clusters[i]] += 1

    # aggregate movie ratings and appearance
    # The csv has name,movie name,rating format; we ignore movie name
    # Convert to numpy as it is easier to iterate
    df = pd.read_csv(ratingCsvLoc)
    actor_names = df.iloc[:, 0].to_numpy()
    each_rating = df.iloc[:, 2].to_numpy()

    for i in range(actor_names.size):
        # Check if this is the first time the actor appears
        if actor_names[i] not in result_ratings:
            result_ratings[actor_names[i]] = 0
            result_ratings_appearance[actor_names[i]] = 0

        result_ratings[actor_names[i]] += each_rating[i]
        result_ratings_appearance[actor_names[i]] += 1

    return result_clusters, result_ratings, result_ratings_appearance
