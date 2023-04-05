#!/usr/bin/env python3
"""
Responsible for creating and processing of data structures and saving numpy arrays to files.
"""
import numpy as np
import pandas as pd

def convertTensors(actors: list, all_vectors: list):
    # Change vectors to a pandas dataframe
    x = []
    for eachVecList in all_vectors:
        y = []
        for eachTensor in eachVecList:
            y.append(eachTensor.tolist())  # this converts the list of tensors to a regular list of lists
        x.append(y)

    return x

def unrollVecAndSave(all_actors: list, all_vectors: list, actors_loc, vectors_loc):
    # Change vectors to a numpy array, no saving
    unrolled_actors = []
    unrolled_vectors = []
    for i in range(len(all_vectors)):
        eachVecList = all_vectors[i]
        for j in range(len(eachVecList)):
            unrolled_actors.append(all_actors[i])
            unrolled_vectors.append(eachVecList[j])

    unrolled_actors = np.array(unrolled_actors)
    unrolled_vectors = np.array(unrolled_vectors)

    saveActorAndVectors(unrolled_actors, unrolled_vectors, actors_loc, vectors_loc)

    return unrolled_actors, unrolled_vectors

def saveActorAndVectors(actors, vectors, actors_loc, vectors_loc):
    np.save(actors_loc, actors)
    np.save(vectors_loc, vectors)
