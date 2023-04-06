#!/usr/bin/env python3
"""
This is the executable file that only embeds the training set and saves into a file after preprocessing.
This can be used separately if you wish to update the embeddings for the training set.
"""
import json
from transformers import BertTokenizer, BertModel
from Logic import preprocess, processList
from Logic.embeddedLearn import embedWords


def wordEmbedTrainingData(model, tokenizer, trainDataLocs):
    roleDescriptionLoc = trainDataLocs[0]
    trainActorsLoc = trainDataLocs[1]
    trainVectorsLoc = trainDataLocs[2]
    trainActorCountsLoc = trainDataLocs[3]

    # embed words for training with pre-trained BERT model and count the appearances of the actor in role descriptions
    train_actors, train_subwords, train_vectors, actor_counts = \
        embedWords(roleDescriptionLoc, model, tokenizer)
    # Remove stop words from the embeddings and get it back
    _, up_train_vectors = preprocess.eliminateStopWords(train_subwords, train_vectors)
    # train_vectors are tensors; convert to a regular list. It will be a 2D list.
    up_train_vectors = processList.convertTensors(train_actors, up_train_vectors)

    # As clustering takes 1D numpy array, the 2D list for vectors needs to be unrolled.
    # Unroll 2D list for vectors with matching actor names and save as a file for future reusability
    unroll_train_actors, train_vec_numpy = processList.unrollVecAndSave\
        (train_actors, up_train_vectors, trainActorsLoc, trainVectorsLoc)

    # Save actor counts dictionary as a json file
    with open(trainActorCountsLoc, "w") as f:
        json.dump(actor_counts, f)

    return unroll_train_actors, train_vec_numpy, actor_counts

if __name__ == "__main__":
    # SETUP
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Load pre-trained model tokenizer with a given bert version
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    roleDescriptionLoc = 'Data/TrainData/Roles.csv'
    trainActorsLoc = 'Data/TrainData/trainActors.npy'
    trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
    trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'

    trainDataLocs = (roleDescriptionLoc, trainActorsLoc, trainVectorsLoc, trainActorCountsLoc)
    wordEmbedTrainingData(model, tokenizer, trainDataLocs)
