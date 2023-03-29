#!/usr/bin/env python3
"""
This is the executable file that only embeds the training set.
This can be used separately if you wish to update the embeddings for the training set.
"""

import embeddedLearn
import processList

roleDescriptionLoc = 'Roles.csv'
trainingVectorLoc = 'trainVectors.npy'
stopWordsLoc = ''

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = embeddedLearn.embedWords(roleDescriptionLoc, 'bert-base-uncased')
processList.tensorsToNumpy(train_actors, train_subwords, train_vectors, trainingVectorLoc, stopWordsLoc)
