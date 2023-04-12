#!/usr/bin/env python3
"""
This is the executable file that embeds the testing/evaluation set,
clustering with the training set from the savefile and generating rank afterwards.
This can be used separately if you wish to re-use your training set's embeddings with new inputs.
"""

import json
import numpy as np
from Logic import preprocess, processList, clustering, actorInfoGeneration, extractTerms, generateRanking
from Logic.embeddedLearn import embedWords
from Logic.setupModel import setupBert


def kmeansCluster(train_vec_numpy, input_vector, n_clusters):
    cluster_vectors = np.concatenate((train_vec_numpy, np.array(input_vector)))
    cluster_data = clustering.kmeansClustering(cluster_vectors, n_clusters)

    return cluster_data

def dbscanCluster(train_vec_numpy, input_vector, eps, min_samples):
    cluster_vectors = np.concatenate((train_vec_numpy, np.array(input_vector)))
    cluster_data = clustering.dbscanClustering(cluster_vectors, eps, min_samples)

    return cluster_data

def scanCluster(clusteringType: str, train_vec_numpy, input_vector, eps=12.2, min_samples=5, n_clusters=25):
    if clusteringType.lower() == "dbscan":
        return dbscanCluster(train_vec_numpy, input_vector, eps, min_samples)
    else:
        return kmeansCluster(train_vec_numpy, input_vector, n_clusters)

def clusterToRankGen(input_actors, up_input_subwords, up_input_vectors):
    # CLUSTERING TO RANKING GENERATION
    # Steps:
    # 1. Get embeddings of a single role description
    # 2. Cluster with the trained data
    # 3. Extract query terms
    # 4. Using the query terms, generate ranking and recommend the first n actors
    # 5. If the actor name for the input data is one of the top n actors, we have a match
    # 6. Loop Steps 1-4 for each role description separately, so that input data do not cluster against one another

    # Load the trained data saved; they are already unrolled
    unroll_train_actors = np.load(trainActorsLoc)
    train_vec_numpy = np.load(trainVectorsLoc)
    # Open saved actor counts as dictionary
    with open(trainActorCountsLoc, 'r') as f:
        appearances = json.load(f)

    total_counts = 0
    for actor in appearances:
        total_counts += appearances[actor]

    numMatch = 0  # number of times the actor name provided as the output in the testing data was predicted
    for i in range(len(input_actors)):
        role_subwords = up_input_subwords[i]

        # CLUSTERING with DBSCAN; remove all border points after
        cluster_data = scanCluster("dbscan", train_vec_numpy, up_input_vectors[i], eps=12.2, min_samples=5)
        role_subwords, cluster_data = preprocess.eliminateBorderPoints(role_subwords, cluster_data.tolist())

        # CLUSTERING with K-means
        #cluster_data = scanCluster("kmeans", train_vec_numpy, up_input_vectors[i], n_clusters=40)

        # ACTOR INFORMATION GENERATION
        # Done in this step now that the clustering data has been obtained
        result_clusters, result_ratings, result_ratings_appearance = \
            actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors,
                                                                       movieRatingLoc)

        try:
            # QUERY EXTRACTION
            input_DF = extractTerms.combine_input_cluster(up_input_subwords[i], cluster_data)
            query_result = extractTerms.extractTerms(k=10, df=input_DF)
            query_clusters = [x[1] for x in query_result]  # list comprehension to make a list of clusters only
            print(query_clusters)

            # RANKING GENERATION WITHOUT NORMALIZATION
            #        top_actor_list = generateRanking.generateRanking \
            #            (query_clusters, result_clusters, actor_counts, result_ratings, result_ratings_appearance, 5)

            # RANKING GENERATION WITH RATIO
            topNum = 7  # Number of top actors to recommend
            w1 = 5  # Weight for similarity
            w2 = 1  # Weight for popularity
            w3 = 1  # Weight for rating
            top_actor_list = generateRanking.generateRankingWithRatio \
                (query_clusters, result_clusters, appearances, total_counts, result_ratings, result_ratings_appearance,
                 topNum, w1, w2, w3)

            # CHECK IF THE ACTUAL ACTOR WAS IN THE RECOMMENDATION
            print("Recommended actors: ", top_actor_list)
            if input_actors[i] in top_actor_list:
                numMatch += 1
                print("Name found!")
        except:
            print("The input description does not have any word with a cluster!")

    return numMatch

def wordEmbedInputData(model, tokenizer, roleDescriptionLoc):
    # Get all embeddings for all input role descriptions, and remove stop words from all of them
    # embed words for testing with pre-trained BERT model
    input_actors, input_subwords, input_vectors, _ = \
        embedWords(roleDescriptionLoc, model, tokenizer)
    # input_vectors are tensors; convert to a regular list. It will be a 2D list.
    up_input_vectors = processList.convertTensors(input_actors, input_vectors)

    # Vectors will be returned as a 2D list, as each element is a role description for a possibly different actor
    # and should be used separately.
    return input_actors, input_subwords, up_input_vectors

def plotDBSCANResult(actor_vectors):
    # Plot DBSCAN result with a given set of hyperparameters.
    # This function is used to find the optimal hyperparameters.
    train_vec_numpy = np.load(trainVectorsLoc)
    cluster_data = scanCluster("dbscan", train_vec_numpy, actor_vectors)

if __name__ == "__main__":
    movieRatingLoc = 'Data/TrainData/MoviesManual.csv'
    trainActorsLoc = 'Data/TrainData/trainActors.npy'
    trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
    trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'
    inputRoleDescriptionLoc = 'Data/TestData/InputDescriptionManual.csv'

    # SETUP pre-trained BERT model with tokenizer
    model, tokenizer = setupBert()

    # WORD EMBEDDING FOR INPUT DATA
    input_actors, up_input_subwords, up_input_vectors = \
        wordEmbedInputData(model, tokenizer, inputRoleDescriptionLoc)

    # CLUSTER INPUT DATA AND GENERATE RANKS
    # Return the total number of correct predictions
    numMatch = clusterToRankGen(input_actors, up_input_subwords, up_input_vectors)

    # Get the accuracy
    accuracy = numMatch / len(input_actors)
    print("Accuracy: ", accuracy)

    """
    # TEST TO FIND THE BEST HYPERPARAMETERS with the first input
    plotDBSCANResult(up_input_vectors[0])

    """
