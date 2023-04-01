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

roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingDataLoc = 'trainedData.csv'
stopWordsLoc = ''

# load training vector

# embed words used for input with pre-trained BERT model
input_actors, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, 'bert-base-uncased')
# Remove stop words from the embeddings
input_actors, input_vectors = preprocess.eliminateStopWords(None, input_subwords, input_vectors, stopWordsLoc)
input_vectors = processList.inputVectorsToNumpy(input_vectors)

# combine training and input to cluster them together
train_DF = pd.read_csv(trainingDataLoc)
train_actors = train_DF["actors"].to_numpy()
train_vectors = train_DF["vectors"].to_numpy()
cluster_vectors = np.concatenate((train_vectors, input_vectors))

# cluster data
cluster_data = clustering.dbscanClustering(cluster_vectors)

# generate actor information with name, cluster information, and average rating
result_clusters, result_ratings, result_appearance = \
    actorInfoGeneration.createDictionary_ClustersActorsRatings(cluster_data, train_actors, movieRatingLoc)

# TODO: generate ranks
top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)

# TODO: print ranks
