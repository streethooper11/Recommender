#!/usr/bin/env python3
"""
Responsible for creating and processing of data structures and saving numpy arrays to files.
"""
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

def unrollVecAndSave(all_actors: list, all_vectors: list, save_loc):
    # Change vectors to a numpy array, no saving
    unrolled_actors = []
    unrolled_vectors = []
    for i in range(len(all_vectors)):
        eachVecList = all_vectors[i]
        for j in range(len(eachVecList)):
            unrolled_actors.append(all_actors[i])
            unrolled_vectors.append(eachVecList[j])

    if save_loc is not None:
        saveActorAndVectors(unrolled_actors, unrolled_vectors, save_loc)

    return unrolled_actors, unrolled_vectors

def saveActorAndVectors(actors: list, vectors: list, save_loc):
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    d = {'actors': actors, 'vectors': vectors}
    df = pd.DataFrame(data=d)
    df.to_csv(save_loc)
