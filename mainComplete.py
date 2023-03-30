#!/usr/bin/env python3
"""
This is the executable file that goes over the whole process from web scraping to ranking generation.
This will run everything regardless of whether the training set was updated or not.
If you wish to continue from embedding the training set run mainPart3_inputClusterAndRankingGenerator.py
"""

import numpy as np
import embeddedLearn
import clustering
import preprocess
import generateRanking

roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingVectorLoc = 'trainVectors.npy'
inputVectorLoc = 'inputVectors.npy'
stopWordsLoc = ''

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = embeddedLearn.embedWords(roleDescriptionLoc, 'bert-base-uncased')
preprocess.tensorsToNumpy(train_subwords, train_vectors, trainingVectorLoc, stopWordsLoc)

# embed words used for input with pre-trained BERT model
_, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, 'bert-base-uncased')
preprocess.tensorsToNumpy(input_subwords, input_vectors, inputVectorLoc, stopWordsLoc)

# combine training and input to cluster them together
cluster_tensors = np.concatenate((np.load(trainingVectorLoc), np.load(inputVectorLoc)))

# cluster data
cluster_data = clustering.dbscanClustering(cluster_tensors)

# generate ranks
top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)

# TODO: print? ranks

