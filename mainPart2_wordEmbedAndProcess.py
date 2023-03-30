#!/usr/bin/env python3
"""
This is the executable file that only embeds the training set and saves into a file after preprocessing.
This can be used separately if you wish to update the embeddings for the training set.
"""

import embeddedLearn
import preprocess
import processList

roleDescriptionLoc = 'Roles.csv'
trainingDataLoc = 'trainedData.csv'
stopWordsLoc = ''

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = embeddedLearn.embedWords(roleDescriptionLoc, 'bert-base-uncased')
# Remove stop words from the embeddings and get it back with updated actor data
train_actors, train_vectors = preprocess.eliminateStopWords(train_actors, train_subwords, train_vectors, stopWordsLoc)
# train_vectors are tensors; convert to numpy, so it can be used in the pre-trained BERT model, and save the file
processList.tensorsToDF(train_actors, train_vectors, trainingDataLoc)
