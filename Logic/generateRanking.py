#!/usr/bin/env python3
"""
File responsible for creating actor ranking for the recommendation system.
"""

def calculateSimilarity(query_clusters, clusters, actor):
    result = 0
    for query_cluster in query_clusters:
        if (query_cluster >= 1) and (query_cluster in clusters[actor]):
            result += clusters[actor][query_cluster]

    return float(result)

def calculatePopularity(appearances, actor):
    if actor not in appearances:
        return 0

    return float(appearances[actor])

def calculateRating(ratings, rating_appearances, actor):
    if actor not in ratings:
        return 0

    return ratings[actor] / float(rating_appearances[actor])

def generateRanking(query_clusters, clusters, appearances, ratings, rating_appearances, topNum=5):
    result = []
    # Weights differ quite a lot because they have different domains eg. popularity can be high so weight is much lower
    w1 = 0.1 # Weight for similarity
    w2 = 0.02 # Weight for popularity
    w3 = 0.05 # Weight for actor

    for actor in clusters:
        similarityScore = calculateSimilarity(query_clusters, clusters, actor) ** w1
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


def calculateSimilarityRatio(query_clusters, clusters, appearances, actor):
    result = 0
    for query_cluster in query_clusters:
        if (query_cluster >= 1) and (query_cluster in clusters[actor]):
            result += clusters[actor][query_cluster]

    # Get a ratio of the result and the total number of cluster counts the actor got throughout all clusters
    return result / float(appearances[actor])

def calculatePopularityRatio(appearances, total_counts,actor):
    if len(appearances) == 0:
        return 0

    # Divide the total number of counts for the actor by the total number of counts for all the actors
    # To get how often the actor appears in role descriptions
    return appearances[actor] / float(total_counts)

def calculateRatingRatio(ratings, rating_appearances, actor):
    if actor not in ratings:
        return 0

    averageRating = ratings[actor] / float(rating_appearances[actor])

    # Average Rating is 0 to 10; divide by 10 to normalize it and get a value between 0 and 1
    return averageRating / 10

def generateRankingWithRatio(query_clusters, clusters, appearances, total_counts, ratings, rating_appearances, topNum=5):
    # This is another way of generating rank, with all 3 features using a ratio value between 0 and 1
    # Weights don't need to differ as much as they are all "normalized"
    result = []
    w1 = 0.7 # Weight for similarity
    w2 = 0.1 # Weight for popularity
    w3 = 0.3 # Weight for rating

    for actor in clusters:
        similarityRatio = calculateSimilarityRatio(query_clusters, clusters, appearances, actor) ** w1
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
