#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
"""
import numpy as np


def eliminateStopWords(actors: list[str], subwords: list[str], vectors: list, stopWordsLoc):
    # Read all stopwords by splitting them with whitespaces
#    stopwords = set(open(stopWordsLoc).read().split())
    # used to remove tokens and some special characters that may appear in descriptions
    remove_words = {"[CLS]", "[SEP]", ",", '"', "'", ";", ":", "!", "$", "^", "@"}

#    stopwords.update(remove_words)  # Combine the two sets

    if actors is None:
        # no actor information; this is input data. Only remove vector data
        # reverse loop to make sure there are no problems when deleting elements
        for i in range(len(subwords) - 1, -1, -1):
            if (subwords[i] in remove_words) or (len(subwords[i]) <= 1):
                vectors.pop(i)
    else:
        # remove actor data and vector data in the index in which the subwords element is a stop word
        # reverse loop to make sure there are no problems when deleting elements
        for i in range(len(subwords) - 1, -1, -1):
            if (subwords[i] in remove_words) or (len(subwords[i]) <= 1):
                actors.pop(i)
                vectors.pop(i)

    return actors, vectors
