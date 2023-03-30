#!/usr/bin/env python3
"""
This is the executable file that embeds the testing/evaluation set,
clustering with the training set from the savefile and generating rank afterwards.
This can be used separately if you wish to re-use your training set's embeddings with new inputs.
"""

import numpy as np
import embeddedLearn
import clustering
import preprocess
import generateRanking

roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingDataLoc = 'trainedData.csv'
stopWordsLoc = ''

# load training vector

# embed words used for input with pre-trained BERT model
_, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, 'bert-base-uncased')
# Remove stop words from the embeddings
_, input_vectors = preprocess.eliminateStopWords(None, input_subwords, input_vectors, stopWordsLoc)

# combine training and input to cluster them together
cluster_tensors = np.concatenate((np.load(trainingDataLoc), input_vectors))

# cluster data
cluster_data = clustering.dbscanClustering(cluster_tensors)

# generate ranks
top_actor_list = generateRanking(cluster_data, train_actors, 5)

# TODO: print? ranks
