#!/usr/bin/env python3
"""
Responsible for preprocessing such as eliminating words. Also saves the result into a file
Source of NLTK stopwords: http://www.nltk.org/nltk_data/ Under section 74 "Stopwords Corpus"

NOTE: This downloads the list of stopwords to the user's computer.
"""

import string

def eliminateStopWords(all_subwords: list, all_vectors: list):
    bert_tokens = ["[CLS]", "[SEP]"]
    with open("Data/NLTK_stopwords.txt", "r") as f:
        nltk = f.readlines()

    nltk_list = [line.strip() for line in nltk]

    stop_words = set(nltk_list + list(string.punctuation) + bert_tokens)

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
