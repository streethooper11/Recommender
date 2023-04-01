#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
"""
import numpy as np


def eliminateStopWords(all_subwords: list, all_vectors: list, stopWordsLoc):
    # Read all stopwords by splitting them with whitespaces
#    stopwords = set(open(stopWordsLoc).read().split())
    # used to remove tokens and some special characters that may appear in descriptions
    remove_words = {"[CLS]", "[SEP]", ",", '"', "'", ";", ":", "!", "$", "^", "@"}

#    stopwords.update(remove_words)  # Combine the two sets

    for i in range(len(all_subwords)):
        each_subword_list = all_subwords[i]
        each_vector_list = all_vectors[i]

        # remove vector data in the index in which the subwords element is a stop word
        # reverse loop to make sure there are no problems when deleting elements
        for j in range(len(each_subword_list) - 1, -1, -1):
            if (each_subword_list[j] in remove_words) or (len(each_subword_list[j]) <= 1):
                each_vector_list.pop(j)

    return all_vectors
