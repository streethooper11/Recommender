#!/usr/bin/env python3
"""
File responsible for creating actor ranking for the recommendation system.
"""

def calculateSimilarity(query_clusters, clusters, appearances, actor):
    result = 0
    for query_cluster in query_clusters:
        if (query_cluster >= 1) and (query_cluster in clusters[actor]):
            result += clusters[actor][query_cluster]

    # The result is divided by the number of words that the actor got in order to normalize it
    return result / float(appearances[actor])

def calculatePopularity(appearances, actor):
    if actor not in appearances:
        return 0

    return appearances[actor]

def calculateRating(ratings, rating_appearances, actor):
    if actor not in ratings:
        return 0

    return ratings[actor] / (rating_appearances[actor])

def generateRanking(query_clusters, clusters, appearances, ratings, rating_appearances, topNum=5):
    result = []
    w1 = 0.1 # As similarityScore is a number between 0 and 1, smaller value makes it more significant
    w2 = 0.02 # As the total number of words will be high, set this low
    w3 = 0.05 # Rating is between 0 and 10

    for actor in clusters:
        similarityScore = calculateSimilarity(query_clusters, clusters, appearances, actor) ** w1
        popularityScore = calculatePopularity(appearances, actor) ** w2
        ratingScore = calculateRating(ratings, rating_appearances, actor) ** w3
        actorScore = similarityScore * popularityScore * ratingScore
        result.append((actor, actorScore))

    # Sort the result by the second value, which is actor score, in reverse order
    result = sorted(result, key=lambda x: x[1], reverse=True)
    result = [x[0] for x in result] # list comprehension to make a list of actor names only

    if len(result) <= topNum:
        return result
    else:
        return result[:topNum]
