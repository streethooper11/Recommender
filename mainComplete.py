#!/usr/bin/env python3
"""
This is the executable file that goes over the whole process from web scraping to ranking generation.
This will run everything regardless of whether the training set was updated or not.
If you wish to continue from embedding the training set run mainPart3_inputClusterAndRankingGenerator.py
"""

import pandas as pd
import numpy as np
import embeddedLearn
import clustering
import preprocess
import actorInfoGeneration
import generateRanking
import processList
from transformers import BertTokenizer, BertModel


roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingDataLoc = 'trainedData.csv'
stopWordsLoc = 'Roles.csv'

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Load pre-trained model tokenizer with a given bert version
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = embeddedLearn.embedWords(roleDescriptionLoc, model, tokenizer)
# Remove stop words from the embeddings and get it back
up_train_vectors = preprocess.eliminateStopWords(train_subwords, train_vectors, stopWordsLoc)
# train_vectors are tensors; convert to a regular list, and save the file. Return the vector 2D list
up_train_vectors = processList.convertTensors(train_actors, up_train_vectors, trainingDataLoc)

# As clustering takes 1D numpy array, the 2D list for vectorsneeds to be unrolled
# At the same time, each vector will be converted to numpy array
unroll_train_actors, train_vec_numpy = processList.unrollVecAndNumpy(train_actors, up_train_vectors)


# We work with each input role description separately, but embeddings can be done as a whole
# Get all embeddings for all input role descriptions, and remove stop words from all of them
input_actors, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, model, tokenizer)
up_input_vectors = preprocess.eliminateStopWords(input_subwords, input_vectors, stopWordsLoc)
up_input_vectors = processList.convertTensors(input_actors, up_input_vectors, None)

# Steps:
# 1. Get embeddings of a single role description
# 2. Cluster with the trained data
# 3. Extract query terms
# 4. Using the query terms, generate ranking and recommend the first n actors
# 5. If the actor name for the input data is one of the top n actors, we have a match
# 6. Loop Steps 1-4 for each role description separately, so that input data do not cluster against one another
numMatch = 0 # number of times the actor name provided as the output in the testing data was predicted
for i in len(input_actors):
    unroll_input_actor, input_vec_numpy = processList.unrollVecAndNumpy(input_actors[i], up_input_vectors[i])
    cluster_vectors = np.concatenate(train_vec_numpy, input_vec_numpy)
    cluster_data = clustering.dbscanClustering(cluster_vectors)

    result_clusters, result_ratings, result_appearance = \
        actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

    top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)
    print(top_actor_list)
    if input_actors[i] in top_actor_list:
        numMatch += 1

# Get the accuracy
accuracy = numMatch / len(input_actors)
print(accuracy)
