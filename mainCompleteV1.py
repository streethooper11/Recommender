#!/usr/bin/env python3
"""
This is the executable file that goes over the process from word embedding to ranking generation
This will run everything regardless of whether the training set was updated or not.
If you wish to include webscraping part run mainCompleteV2.py instead.
If you wish to re-use the embedded training set run with a new testing data
run mainPart3_inputClusterAndRankingGenerator.py instead.
"""

import json
import numpy as np
import embeddedLearn
import clustering
import extractTerms
import preprocess
import actorInfoGeneration
import generateRanking
import processList
from transformers import BertTokenizer, BertModel


roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'Roles.csv'
trainVectorsLoc = 'trainVectors.csv'
trainActorCountsLoc = 'trainActorCounts.json'
testingDataloc = 'testingData.csv'
stopWordsLoc = ''


# SETUP
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Load pre-trained model tokenizer with a given bert version
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# PART 2: WORD EMBEDDING FOR TRAINING DATA
# embed words for training with pre-trained BERT model and count the appearances of the actor in role descriptions
train_actors, train_subwords, train_vectors, actor_counts = \
    embeddedLearn.embedWords(roleDescriptionLoc, model, tokenizer)
# Remove stop words from the embeddings and get it back
_, up_train_vectors = preprocess.eliminateStopWords(train_subwords, train_vectors, stopWordsLoc)
# train_vectors are tensors; convert to a regular list. It will be a 2D list.
up_train_vectors = processList.convertTensors(train_actors, up_train_vectors)

# As clustering takes 1D numpy array, the 2D list for vectors needs to be unrolled.
# Unroll 2D list for vectors with matching actor names and save as a file for future reusability
unroll_train_actors, train_vec = processList.unrollVecAndSave(train_actors, up_train_vectors, trainVectorsLoc)
train_vec_numpy = np.array(train_vec)

# Save actor counts dictionary as a json file for safety
with open(trainActorCountsLoc, "w") as f:
    json.dump(actor_counts, f)


# PART 3: WORD EMBEDDING FOR INPUT DATA
# We work with each input role description separately, but embeddings can be done as a whole
# Get all embeddings for all input role descriptions, and remove stop words from all of them
input_actors, input_subwords, input_vectors, _ = embeddedLearn.embedWords(inputRoleDescriptionLoc, model, tokenizer)
# Remove stop words from the embeddings and get it back
up_input_subwords, up_input_vectors = preprocess.eliminateStopWords(input_subwords, input_vectors, stopWordsLoc)
# Convert tensors to a regular 2D list and get it back
# Input data will not be unrolled as each element is a role description and should be used as one.
up_input_vectors = processList.convertTensors(input_actors, up_input_vectors)
# Save for testing purposes to see if it works correctly
processList.saveActorAndVectors(input_actors, up_input_vectors, testingDataloc)


# PART 4-7: CLUSTERING TO RANKING GENERATION
# Steps:
# 1. Get embeddings of a single role description
# 2. Cluster with the trained data
# 3. Extract query terms
# 4. Using the query terms, generate ranking and recommend the first n actors
# 5. If the actor name for the input data is one of the top n actors, we have a match
# 6. Loop Steps 1-4 for each role description separately, so that input data do not cluster against one another
numMatch = 0 # number of times the actor name provided as the output in the testing data was predicted
for i in range(len(input_actors)):
    actor_name = input_actors[i]

    cluster_vectors = np.concatenate((train_vec_numpy, np.array(up_input_vectors[i])))
    cluster_data = clustering.dbscanClustering(cluster_vectors)

    result_clusters, result_ratings, result_ratings_appearance = \
        actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

    input_DF = extractTerms.combine_input_cluster(up_input_subwords[i], cluster_data)
    query_result = extractTerms.extractTerms(k=5, df=input_DF)
    query_clusters = [x[1] for x in query_result] # list comprehension to make a list of clusters only

    print(query_clusters) # test

    top_actor_list = generateRanking.generateRanking\
        (query_clusters, result_clusters, actor_counts, result_ratings, result_ratings_appearance, 5)

    print("Recommended actors: ", top_actor_list)
    if actor_name in top_actor_list:
        numMatch += 1
        print("Name found!") # test

# Get the accuracy
accuracy = numMatch / len(input_actors)
print("Accuracy: ", accuracy)
