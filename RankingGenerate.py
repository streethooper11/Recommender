import ProcessList

def generateRanking(clusters, actors, topNum):
    cluster_actor_dict, index = ProcessList.createDictionary_ClustersAndActors(clusters, actors)

    actor_scores = dict()

    # Similarity score
    while index < len(clusters):

    # Popularity

    # Rating

    # Calculate score

    # Return top actors/actresses based on the argument given
    if len(actor_scores) < topNum:
        return sorted(actor_scores, key=actors.get, reverse=True)
    else:
        return sorted(actor_scores, key=actors.get, reverse=True)[:topNum]
