#!/usr/bin/env python3
"""
This is the executable file that goes over the whole process from web scraping to ranking generation
This will run everything regardless of whether the training set was updated or not.

If you wish to update the training set with new embeddings run part2_trainDataEmbed.py only
If you wish to re-use the embedded training set run part3_inputDataEmbedToRank.py only
"""

from Logic import actorInfoGeneration, extractTerms, generateRanking
from Logic.setupModel import setupBert
from part1_webscrapeInputData import webscrapeInputData
from part1_webscrapeTrainData import webscrapeTrainData
from part2_trainDataEmbed import wordEmbedTrainingData
from part3_inputDataEmbedToRank import wordEmbedInputData, scanCluster

roleDescriptionLoc = 'Data/TrainData/Roles.csv'
movieRatingLoc = 'Data/TrainData/Movies.csv'
trainActorsLoc = 'Data/TrainData/trainActors.npy'
trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'
inputRoleDescriptionLoc = 'Data/TestData/InputDescription.csv'

# PART 1: DATA COLLECTION
webscrapeTrainData()
webscrapeInputData()

# SETUP pre-trained BERT model with tokenizer
model, tokenizer = setupBert()

# PART 2: WORD EMBEDDING FOR TRAINING DATA
trainDataLocs = (roleDescriptionLoc, trainActorsLoc, trainVectorsLoc, trainActorCountsLoc)
unroll_train_actors, train_vec_numpy, appearances = \
    wordEmbedTrainingData(model, tokenizer, trainDataLocs)

# PART 3: WORD EMBEDDING FOR INPUT DATA
input_actors, up_input_subwords, up_input_vectors = \
    wordEmbedInputData(model, tokenizer, inputRoleDescriptionLoc)

# PART 4-7: CLUSTERING TO RANKING GENERATION
# Steps:
# 1. Retrieve embeddings of a single role description and cluster with the trained data
# 2. Generate the actor information about the training data along with the cluster results
# 3. Extract query terms
# 4. Using the query terms, generate ranking and recommend the first n actors
# 5. If the actor name for the input data is one of the top n actors, we have a match
# 6. Loop Steps 1-5 for each role description separately, so that input data do not cluster against one another
# WORD EMBEDDING FOR INPUT DATA
input_actors, up_input_subwords, up_input_vectors = \
    wordEmbedInputData(model, tokenizer, inputRoleDescriptionLoc)

numMatch = 0  # number of times the actor name provided as the output in the testing data was predicted
for i in range(len(input_actors)):
    # CLUSTERING
    cluster_data = scanCluster("dbscan", train_vec_numpy, up_input_vectors[i])
    # cluster_data = scanCluster("kmeans", train_vec_numpy, up_input_vectors[i])

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
        (query_clusters, result_clusters, appearances, result_ratings, result_ratings_appearance, 5)

    # CHECK IF THE ACTUAL ACTOR WAS IN THE RECOMMENDATION
    print("Recommended actors: ", top_actor_list)
    actor_name = input_actors[i]
    if actor_name in top_actor_list:
        numMatch += 1
        print("Name found!")

# PRINT THE ACCURACY
accuracy = numMatch / len(input_actors)
print("Accuracy: ", accuracy)
