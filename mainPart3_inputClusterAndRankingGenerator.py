#!/usr/bin/env python3
"""
This is the executable file that embeds the testing/evaluation set,
clustering with the training set from the savefile and generating rank afterwards.
This can be used separately if you wish to re-use your training set's embeddings with new inputs.
"""

import numpy as np
import pandas as pd
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
stopWordsLoc = ''

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Load pre-trained model tokenizer with a given bert version
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained data saved
df = pd.read_csv(stopWordsLoc)
train_actors = df.iloc[:, 0].tolist()
up_train_vectors = df.iloc[:, 1].tolist()

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
