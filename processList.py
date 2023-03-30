#!/usr/bin/env python3
"""
Responsible for creating and processing of data structures and saving numpy arrays to files.
"""
import pandas as pd
import numpy as np


def tensorsToDF(actors: list, vectors, save_loc):
    # Change vectors to a pandas dataframe
    x = []
    for eachVector in vectors:
        x.append(eachVector.tolist())  # this converts the tensor to a regular list

    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    d = {'actors': actors, 'vectors': x}
    df = pd.DataFrame(data=d)
    df.to_csv(save_loc)

    return df


def inputVectorsToNumpy(vectors):
    # Change vectors to a numpy array, no saving
    x = []
    for eachVector in vectors:
        x.append(eachVector.tolist())  # this converts the tensor to a regular list

    x = np.array(x)

    return x
