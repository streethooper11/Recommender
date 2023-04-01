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
stopWordsLoc = ''

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


# Input will be different as we should not read all the content.
# Instead, it will do the following:
# 1. Embed words of all role descriptions in the file
# 2. Preprocess all the output
# 3. Get embeddings of a single role description
# 4. Cluster with the trained data
# 5. Loop Steps 3 and 4 for each role description, so that input data do not cluster against one another
# input_actors, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, model, tokenizer)
# up_input_vectors = preprocess.eliminateStopWords(input_subwords, input_vectors, stopWordsLoc)
# up_input_vectors = processList.convertTensors(input_actors, up_input_vectors, None)
# for i in len(input_actors):
#     unroll_input_actor, input_vec_numpy = processList.unrollVecAndNumpy(input_actors[i], up_input_vectors[i])

#     combine training and input to cluster them together
#     cluster_vectors = np.concatenate(train_vec_numpy, input_vec_numpy)
#     cluster data
#     cluster_data = clustering.dbscanClustering(cluster_vectors)

#     generate actor information with name, cluster information, and average rating
#     result_clusters, result_ratings, result_appearance = \
#         actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

#     top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)
#     print(top_actor_list)
#


cluster_vectors = train_vec_numpy

# cluster data
cluster_data = clustering.dbscanClustering(cluster_vectors)

# generate actor information with name, cluster information, and average rating
result_clusters, result_ratings, result_appearance = \
    actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, unroll_train_actors, movieRatingLoc)

# TODO: generate ranks
top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)

# TODO: print ranks
print(top_actor_list)

# TODO: accuracy