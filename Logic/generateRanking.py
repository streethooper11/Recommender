#!/usr/bin/env python3
"""
File responsible for creating actor ranking for the recommendation system.
"""

def calculateSimilarity(query_clusters, clusters, role_appearances, actor):
    result = 0
    for query_cluster in query_clusters:
        if (query_cluster >= 1) and (query_cluster in clusters[actor]):
            result += clusters[actor][query_cluster]

    # The result is divided by the number of role appearances to normalize it
    return result / float(role_appearances[actor])

def calculatePopularity(role_appearances, actor):
    if actor not in role_appearances:
        return 0

    return role_appearances[actor]

def calculateRating(ratings, rating_appearances, actor):
    if actor not in ratings:
        return 0

    return ratings[actor] / (rating_appearances[actor])

def generateRanking(query_clusters, clusters, role_appearances, ratings, rating_appearances, topNum=5):
    result = []
    w1 = 0.5
    w2 = 0.05
    w3 = 0.1

    for actor in clusters:
        similarityScore = calculateSimilarity(query_clusters, clusters, role_appearances, actor) ** w1
        popularityScore = calculatePopularity(role_appearances, actor) ** w2
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
