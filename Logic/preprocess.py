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
