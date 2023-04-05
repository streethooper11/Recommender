#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
Use of NLTK stopwords from https://pythonspot.com/nltk-stop-words/

NOTE: This downloads the list of stopwords to the user's computer.
"""
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords') # Download stopwords


def eliminateStopWords(all_subwords: list, all_vectors: list):
    bert_tokens = ["[CLS]", "[SEP]"]
    stop_words = set(stopwords.words("english") + list(string.punctuation) + bert_tokens)

    for i in range(len(all_subwords)):
        each_subword_list = all_subwords[i]
        each_vector_list = all_vectors[i]

        # remove vector data in the index in which the subwords element is a stop word
        # reverse loop to make sure there are no problems when deleting elements
        for j in range(len(each_subword_list) - 1, -1, -1):
            subword = each_subword_list[j]

            if (subword in stop_words) or (len(subword) <= 1) or (subword.isdigit()):
                each_subword_list.pop(j)
                each_vector_list.pop(j)

    return all_subwords, all_vectors
