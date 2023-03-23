import numpy as np
import EmbeddedLearn
import Clustering
import ProcessList
import RankingGenerate

csvLoc = 'output.csv'
vectorLoc = 'trainVectors.npy'
inputCsvLoc = 'input.csv'
inputLoc = 'inputVectors.npy'
stopWordsLoc = ''

# embed words for training with pre-trained BERT model
train_actors, train_subwords, train_vectors = EmbeddedLearn.embedWords(csvLoc, 'bert-base-uncased')
ProcessList.tensorsToNumpy(train_subwords, train_vectors, vectorLoc, stopWordsLoc)

# embed words used for input with pre-trained BERT model
_, input_subwords, input_vectors = EmbeddedLearn.embedWords(inputCsvLoc, 'bert-base-uncased')
ProcessList.tensorsToNumpy(input_subwords, input_vectors, inputLoc, stopWordsLoc)

# combine training and input to cluster them together
cluster_tensors = np.concatenate((np.load(vectorLoc), np.load(inputLoc)))

# cluster data
cluster_data = Clustering.dbscanClustering(cluster_tensors)

# generate ranks
top_actor_list = RankingGenerate.generateRanking(cluster_data, train_actors, 5)

# TODO: print? ranks
