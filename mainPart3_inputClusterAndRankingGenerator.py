#!/usr/bin/env python3
"""
This is the executable file that embeds the testing/evaluation set,
clustering with the training set and generating rank afterwards.
This can be used separately if you wish to re-use your training set's embeddings with new inputs.
"""

import numpy as np
import embeddedLearn
import clustering
import processList
import generateRanking

roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingVectorLoc = 'trainVectors.npy'
inputVectorLoc = 'inputVectors.npy'
stopWordsLoc = ''

# load training vector

# embed words used for input with pre-trained BERT model
_, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, 'bert-base-uncased')
processList.tensorsToNumpy(input_subwords, input_vectors, inputVectorLoc, stopWordsLoc)

# combine training and input to cluster them together
cluster_tensors = np.concatenate((np.load(trainingVectorLoc), np.load(inputVectorLoc)))

# cluster data
cluster_data = clustering.dbscanClustering(cluster_tensors)

# generate ranks
top_actor_list = generateRanking(cluster_data, train_actors, 5)

# TODO: print? ranks
