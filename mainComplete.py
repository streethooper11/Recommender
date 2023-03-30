#!/usr/bin/env python3
"""
This is the executable file that goes over the whole process from web scraping to ranking generation.
This will run everything regardless of whether the training set was updated or not.
If you wish to continue from embedding the training set run mainPart3_inputClusterAndRankingGenerator.py
"""

import embeddedLearn
import clustering
import preprocess
import generateRanking
import processList

roleDescriptionLoc = 'Roles.csv'
movieRatingLoc = 'Movies.csv'
inputRoleDescriptionLoc = 'input.csv'
trainingDataLoc = 'trainedData.csv'
stopWordsLoc = ''

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = embeddedLearn.embedWords(roleDescriptionLoc, 'bert-base-uncased')
# Remove stop words from the embeddings and get it back with updated actor data
train_actors, train_vectors = preprocess.eliminateStopWords(train_actors, train_subwords, train_vectors, stopWordsLoc)
# train_vectors are tensors; convert to numpy, so it can be used in the pre-trained BERT model, and save the file
train_DF = processList.tensorsToDF(train_actors, train_vectors, trainingDataLoc)

# embed words used for input with pre-trained BERT model
# _, input_subwords, input_vectors = embeddedLearn.embedWords(inputRoleDescriptionLoc, 'bert-base-uncased')
# Remove stop words from the embeddings
# _, input_vectors = preprocess.eliminateStopWords(None, input_subwords, input_vectors, stopWordsLoc)

# combine training and input to cluster them together
# cluster_vectors = np.concatenate((np.load(trainingVecLoc), np.load(inputVecLoc)))
cluster_vectors = train_DF["vectors"].to_numpy()

vector_check = len(cluster_vectors[0])
for each_vector in cluster_vectors:
    if len(each_vector) != vector_check:
        print("FALSE!!!")
        break

# cluster data
cluster_data = clustering.dbscanClustering(cluster_vectors)

print(cluster_data)

# generate ranks
# top_actor_list = generateRanking.generateRanking(cluster_data, train_actors, 5)

# TODO: print? ranks
