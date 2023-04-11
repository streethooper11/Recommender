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


def kmeansCluster(train_vec_numpy, input_vector):
    cluster_vectors = np.concatenate((train_vec_numpy, np.array(input_vector)))
    cluster_data = clustering.kmeansClustering(cluster_vectors)

    return cluster_data

def dbscanCluster(train_vec_numpy, input_vector):
    cluster_vectors = np.concatenate((train_vec_numpy, np.array(input_vector)))
    cluster_data = clustering.dbscanClustering(cluster_vectors)

    return cluster_data

def scanCluster(clusteringType: str, train_vec_numpy, input_vector):
    if clusteringType.lower() == "dbscan":
        return dbscanCluster(train_vec_numpy, input_vector)
    else:
        return kmeansCluster(train_vec_numpy, input_vector)

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
        actor_counts = json.load(f)

    numMatch = 0 # number of times the actor name provided as the output in the testing data was predicted

    for i in range(len(input_actors)):
        actor_name = input_actors[i]

        cluster_data = scanCluster("dbscan", train_vec_numpy, up_input_vectors[i])
        #cluster_data = scanCluster("kmeans", train_vec_numpy, up_input_vectors[i])

        result_clusters, result_ratings, result_ratings_appearance = \
            actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

        input_DF = extractTerms.combine_input_cluster(up_input_subwords[i], cluster_data)
        query_result = extractTerms.extractTerms(k=5, df=input_DF)
        query_clusters = [x[1] for x in query_result]  # list comprehension to make a list of clusters only

        top_actor_list = generateRanking.generateRanking \
            (query_clusters, result_clusters, actor_counts, result_ratings, result_ratings_appearance, 5)

        print("Recommended actors: ", top_actor_list)
        if actor_name in top_actor_list:
            numMatch += 1
            print("Name found!")

    # Get the accuracy
    accuracy = numMatch / len(input_actors)
    print("Accuracy: ", accuracy)

def wordEmbedInputData(model, tokenizer, roleDescriptionLoc):
    # Get all embeddings for all input role descriptions, and remove stop words from all of them
    # embed words for testing with pre-trained BERT model
    input_actors, input_subwords, input_vectors, _ = \
        embedWords(roleDescriptionLoc, model, tokenizer)
    # Remove stop words from the embeddings and get it back
    up_input_subwords, up_input_vectors = preprocess.eliminateStopWords(input_subwords, input_vectors)
    # input_vectors are tensors; convert to a regular list. It will be a 2D list.
    up_input_vectors = processList.convertTensors(input_actors, up_input_vectors)

    # Vectors will be returned as a 2D list, as each element is a role description for a possibly different actor
    # and should be used separately.
    return input_actors, up_input_subwords, up_input_vectors

if __name__ == "__main__":
    movieRatingLoc = 'Data/TrainData/Movies.csv'
    trainActorsLoc = 'Data/TrainData/trainActors.npy'
    trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
    trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'
    inputRoleDescriptionLoc = 'Data/TestData/InputDescription.csv'

    # SETUP pre-trained BERT model with tokenizer
    model, tokenizer = setupBert()

    # Load the trained data saved; they are already unrolled
    unroll_train_actors = np.load(trainActorsLoc)
    train_vec_numpy = np.load(trainVectorsLoc)
    # Open saved actor counts as dictionary
    with open(trainActorCountsLoc, 'r') as f:
        actor_counts = json.load(f)

    # WORD EMBEDDING FOR INPUT DATA
    input_actors, up_input_subwords, up_input_vectors = \
        wordEmbedInputData(model, tokenizer, inputRoleDescriptionLoc)

    numMatch = 0 # number of times the actor name provided as the output in the testing data was predicted
    for i in range(len(input_actors)):
        # CLUSTERING
        # cluster_data = scanCluster("dbscan", train_vec_numpy, up_input_vectors[i])
        cluster_data = scanCluster("kmeans", train_vec_numpy, up_input_vectors[i])

        # ACTOR INFORMATION GENERATION
        # Done in this step now that the clustering data has been obtained
        result_clusters, result_ratings, result_ratings_appearance = \
            actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

        # QUERY EXTRACTION
        input_DF = extractTerms.combine_input_cluster(up_input_subwords[i], cluster_data)
        query_result = extractTerms.extractTerms(k=5, df=input_DF)
        query_clusters = [x[1] for x in query_result]  # list comprehension to make a list of clusters only

        # RANKING GENERATION
        top_actor_list = generateRanking.generateRanking \
            (query_clusters, result_clusters, actor_counts, result_ratings, result_ratings_appearance, 5)

        # CHECK IF THE ACTUAL ACTOR WAS IN THE RECOMMENDATION
        print("Recommended actors: ", top_actor_list)
        actor_name = input_actors[i]
        if actor_name in top_actor_list:
            numMatch += 1
            print("Name found!")

    # PRINT THE ACCURACY
    accuracy = numMatch / len(input_actors)
    print("Accuracy: ", accuracy)
