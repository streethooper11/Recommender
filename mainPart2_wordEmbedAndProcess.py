#!/usr/bin/env python3
"""
This is the executable file that only embeds the training set and saves into a file after preprocessing.
This can be used separately if you wish to update the embeddings for the training set.
"""

import embeddedLearn
import preprocess
import processList
from transformers import BertTokenizer, BertModel

roleDescriptionLoc = 'Roles.csv'
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
# train_vectors are tensors; convert to a regular list, and save the file
_ = processList.convertTensors(train_actors, up_train_vectors, trainingDataLoc)
