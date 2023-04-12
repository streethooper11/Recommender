#!/usr/bin/env python3
"""
This is the executable file that only embeds the training set and saves into a file after preprocessing.
This can be used separately if you wish to update the embeddings for the training set.
"""
import json
from Logic import preprocess, processList
from Logic.embeddedLearn import embedWords
from Logic.setupModel import setupBert


def wordEmbedTrainingData(model, tokenizer, trainDataLocs):
    roleDescriptionLoc = trainDataLocs[0]
    trainActorsLoc = trainDataLocs[1]
    trainVectorsLoc = trainDataLocs[2]
    trainActorCountsLoc = trainDataLocs[3]

    # embed words for training with pre-trained BERT model and count the appearances of the actor in role descriptions
    train_actors, train_subwords, train_vectors, appearances = \
        embedWords(roleDescriptionLoc, model, tokenizer)
    # train_vectors are tensors; convert to a regular list. It will be a 2D list.
    up_train_vectors = processList.convertTensors(train_actors, train_vectors)

    # As clustering takes 1D numpy array, the 2D list for vectors needs to be unrolled.
    # Unroll 2D list for vectors with matching actor names and save as a file for future reusability
    unroll_train_actors, train_vec_numpy = processList.unrollVecAndSave\
        (train_actors, up_train_vectors, trainActorsLoc, trainVectorsLoc)

    # Save actor counts dictionary as a json file
    with open(trainActorCountsLoc, "w") as f:
        json.dump(appearances, f)

    return unroll_train_actors, train_vec_numpy, appearances

if __name__ == "__main__":
    # SETUP pre-trained BERT model with tokenizer
    model, tokenizer = setupBert()

    roleDescriptionLoc = 'Data/TrainData/RolesManual.csv'
    trainActorsLoc = 'Data/TrainData/trainActors.npy'
    trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
    trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'

    trainDataLocs = (roleDescriptionLoc, trainActorsLoc, trainVectorsLoc, trainActorCountsLoc)
    wordEmbedTrainingData(model, tokenizer, trainDataLocs)
