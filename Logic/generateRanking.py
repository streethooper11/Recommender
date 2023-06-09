#!/usr/bin/env python3
"""
File responsible for creating actor ranking for the recommendation system.
"""

def calculateSimilarityRatio(query_clusters, clusters, appearances, actor, numQuery):
    result = 0
    for query_cluster in query_clusters:
        if (query_cluster >= 1) and (query_cluster in clusters[actor]):
            result += clusters[actor][query_cluster]

    # Get the total result, divided by the
    # multiplication of the total number of clusters the actor has and
    # the total number of times the actor was scanned to match the cluster, which is equal to the number of
    # the input clusters
    return result / float(appearances[actor] * numQuery)

def calculatePopularityRatio(appearances, total_counts,actor):
    if len(appearances) == 0:
        return 0

    # Divide the total number of cluster counts for the actor by the total number of cluster counts for all the actors
    # To get how often the actor appears in role descriptions
    return appearances[actor] / float(total_counts)

def calculateRatingRatio(ratings, rating_appearances, actor):
    if actor not in ratings:
        return 0

    averageRating = ratings[actor] / float(rating_appearances[actor])

    # Average Rating is 0 to 10; divide by 10 to normalize it and get a value between 0 and 1
    return averageRating / 10

def generateRankingWithRatio(query_clusters, clusters, appearances, total_counts, ratings, rating_appearances, topNum,
                             w1, w2, w3):
    # This is another way of generating rank, with all 3 features using a ratio value between 0 and 1
    # Weights don't need to differ as much as they are all "normalized"
    result = []

    numQuery = len(query_clusters) # The number of clusters, to be used in similarity ratio
    for actor in clusters:
        similarityRatio = calculateSimilarityRatio(query_clusters, clusters, appearances, actor, numQuery) ** w1
        popularityRatio = calculatePopularityRatio(appearances, total_counts, actor) ** w2
        ratingRatio = calculateRatingRatio(ratings, rating_appearances, actor) ** w3
        actorScore = similarityRatio * popularityRatio * ratingRatio
        result.append((actor, actorScore))

    # Sort the result by the second value, which is actor score, in reverse order
    result = sorted(result, key=lambda x: x[1], reverse=True)
    result = [x[0] for x in result] # list comprehension to make a list of actor names only

    if len(result) <= topNum:
        return result
    else:
        return result[:topNum]
