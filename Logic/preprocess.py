#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
Source of NLTK stopwords: http://www.nltk.org/nltk_data/ Under section 74 "Stopwords Corpus"
"""

def eliminateStopWords(each_subword_list: list, each_vector_list: list, stop_words):
    # remove vector data in the index in which the subwords element is a stop word
    # reverse loop to make sure there are no problems when deleting elements
    for i in range(len(each_subword_list) - 1, -1, -1):
        subword = each_subword_list[i]

        if (subword in stop_words) or (len(subword) <= 1) or (subword.isdigit()):
            each_subword_list.pop(i)
            each_vector_list.pop(i)

    return each_subword_list, each_vector_list

def eliminateBorderPoints(each_subword_list: list, cluster_data: list):
    # remove input clusters of -1 before getting query terms so each query term contributes to similarity score
    # the clusters of input data are the last len(each_subword_list) number of elements
    neg_length = (-1) * len(each_subword_list)
    i = -1

    while i >= neg_length:
        if (cluster_data[i]) == -1:
            each_subword_list.pop(i)
            cluster_data.pop(i)
            neg_length += 1
        else:
            i -= 1

    return each_subword_list, cluster_data
