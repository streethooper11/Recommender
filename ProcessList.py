def createDictionary_ClustersAndActors(clusters, actors):
    """
    Creates a dictionary from a list of clusters and a list of actors

    :param actors: List of actors
    :param clusters: List of clusters
    :return: A dictionary that consists of clusters as keys and a dictionary of actors and count as values
    """

    result = dict()

    for i in range(len(clusters)):
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

    return result
