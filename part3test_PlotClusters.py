#!/usr/bin/env python3
"""
A file that plots the DBSCAN clustering result as a bar graph
"""

import json
import random

import numpy as np
from matplotlib import pyplot as plt

from Logic import preprocess, processList, clustering, actorInfoGeneration, extractTerms, generateRanking
from Logic.embeddedLearn import embedWords
from Logic.setupModel import setupBert


def wordEmbedInputData(model, tokenizer, roleDescriptionLoc):
    # Get all embeddings for all input role descriptions, and remove stop words from all of them
    # embed words for testing with pre-trained BERT model
    input_actors, input_subwords, input_vectors, _ = \
        embedWords(roleDescriptionLoc, model, tokenizer)
    # input_vectors are tensors; convert to a regular list. It will be a 2D list.
    up_input_vectors = processList.convertTensors(input_actors, input_vectors)

    # Vectors will be returned as a 2D list, as each element is a role description for a possibly different actor
    # and should be used separately.
    return input_actors, input_subwords, up_input_vectors

def plotkmeansResult(actor_vectors, n_clusters=10):
    # Plot K-means result with a given k.
    # This function is used to find the optimal k
    train_vec_numpy = np.load(trainVectorsLoc)

    cluster_vectors = np.concatenate((train_vec_numpy, np.array(actor_vectors)))
    cluster_data = clustering.kmeansClustering(cluster_vectors, n_clusters=n_clusters)

    dataToPlot = dict()

    # Count only clustered points
    for i in range(len(cluster_data)):
        if cluster_data[i] not in dataToPlot:
            dataToPlot[cluster_data[i]] = 0
        dataToPlot[cluster_data[i]] += 1

    # Usage of mathplotlib from CPSC 501 knowledge
    plt.bar(range(len(dataToPlot)), list(dataToPlot.values()), align="center")
    plt.xticks(range(len(dataToPlot)), list(dataToPlot.keys()))
    plt.ylim(ymin=0, ymax=6000)
    title = "K-means clusters with counts, k = " + str(n_clusters) + ", all data"
    plt.title(title, fontsize=10)
    plt.show()

    return cluster_data

def plotkmeansInputOnly(input_length, cluster_data, n_clusters=10):
    neg_length = (-1) * input_length
    dataToPlot = dict()

    # Count only clustered point for input
    i = -1
    while i >= neg_length:
        if cluster_data[i] > -1:
            if cluster_data[i] not in dataToPlot:
                dataToPlot[cluster_data[i]] = 0
            dataToPlot[cluster_data[i]] += 1
        i -= 1

    # Usage of mathplotlib from CPSC 501 knowledge
    plt.bar(range(len(dataToPlot)), list(dataToPlot.values()), align="center")
    plt.xticks(range(len(dataToPlot)), list(dataToPlot.keys()))
    plt.ylim(ymin=0, ymax=20)
    title = "K-means clusters with counts, k = " + str(n_clusters) + ", input data"
    plt.title(title, fontsize=10)
    plt.show()

def plotDBSCANResult(actor_vectors, eps=12.2, min_samples=5):
    # Plot DBSCAN result with a given set of hyperparameters.
    # This function is used to find the optimal hyperparameters.
    train_vec_numpy = np.load(trainVectorsLoc)

    cluster_vectors = np.concatenate((train_vec_numpy, np.array(actor_vectors)))
    cluster_data = clustering.dbscanClustering(cluster_vectors, eps=eps, min_samples=min_samples)

    dataToPlot = dict()

    # Count only clustered points
    for i in range(len(cluster_data)):
        if cluster_data[i] > -1:
            if cluster_data[i] not in dataToPlot:
                dataToPlot[cluster_data[i]] = 0
            dataToPlot[cluster_data[i]] += 1

    # Usage of mathplotlib from CPSC 501 knowledge
    plt.bar(range(len(dataToPlot)), list(dataToPlot.values()), align="center")
    plt.xticks(range(len(dataToPlot)), list(dataToPlot.keys()))
    plt.ylim(ymin=0, ymax=1500)
    title = "DBSCAN Clusters with counts, epsilon = " + str(eps) + " and min_samples = " + str(min_samples) + ", all data"
    plt.title(title, fontsize=10)
    plt.show()

    return cluster_data

def plotDBSCANInputOnly(input_length, cluster_data, eps=12.2, min_samples=5):
    neg_length = (-1) * input_length
    dataToPlot = dict()

    # Count only clustered point for input
    i = -1
    while i >= neg_length:
        if cluster_data[i] > -1:
            if cluster_data[i] not in dataToPlot:
                dataToPlot[cluster_data[i]] = 0
            dataToPlot[cluster_data[i]] += 1
        i -= 1

    # Usage of mathplotlib from CPSC 501 knowledge
    plt.bar(range(len(dataToPlot)), list(dataToPlot.values()), align="center")
    plt.xticks(range(len(dataToPlot)), list(dataToPlot.keys()))
    plt.ylim(ymin=0, ymax=50)
    title = "DBSCAN Clusters with counts, epsilon = " + str(eps) + " and min_samples = " + str(min_samples) + ", input data"
    plt.title(title, fontsize=10)
    plt.show()

if __name__ == "__main__":
    movieRatingLoc = 'Data/TrainData/MoviesManual.csv'
    trainActorsLoc = 'Data/TrainData/trainActors.npy'
    trainVectorsLoc = 'Data/TrainData/trainVectors.npy'
    trainActorCountsLoc = 'Data/TrainData/trainActorCounts.json'
    inputRoleDescriptionLoc = 'Data/TestData/InputDescriptionManual.csv'

    # SETUP pre-trained BERT model with tokenizer
    model, tokenizer = setupBert()

    # WORD EMBEDDING FOR INPUT DATA
    input_actors, up_input_subwords, up_input_vectors = \
        wordEmbedInputData(model, tokenizer, inputRoleDescriptionLoc)

    randIndex = random.randint(0, len(up_input_vectors) - 1)

    # TEST TO FIND THE BEST HYPERPARAMETERS
    cluster_data = plotDBSCANResult(up_input_vectors[randIndex], eps=12.5, min_samples=6)
    plotDBSCANInputOnly(len(up_input_vectors[randIndex]), cluster_data, eps=12.5, min_samples=6)
    cluster_data = plotkmeansResult(up_input_vectors[randIndex], n_clusters=40)
    plotkmeansInputOnly(len(up_input_vectors[randIndex]), cluster_data, n_clusters=40)
