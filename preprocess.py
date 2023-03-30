#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
"""
import numpy as np


def eliminateStopWords(actors, subwords, vectors, stopWordsLoc):
    # Read all stopwords by splitting them with whitespaces
    stopwords = set(open(stopWordsLoc).read().split())
    # used to remove tokens and some special characters that may appear in descriptions
    remove_words = {"[CLS]", "[SEP]", ",", '"', "'", ";", ":", "!", "$", "^", "@"}

    # remove tensors for separators, some special characters, and 1-length characters
    info_paragraph = [t for t in zip(subwords, vectors)
                      if (t[0] not in remove_words) and (len(t[0]) > 1)]
    # then split into lists again
    subwords, vectors = zip(*info_paragraph)

    return subwords, vectors
