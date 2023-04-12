#https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
#https://stackoverflow.com/questions/60642043/how-fit-transform-transform-and-tfidfvectorizer-works
#https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python/

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def combine_input_cluster(input_subwords, cluster_data):
    """
    Puts input subwords together with their cluster value into a dataframe 

    :param input_subwords: input subwords array
    :param cluster_data: the result from DBSCAN clustering 
    return final dataframe
    """
    start_index = len(cluster_data) - len(input_subwords)
    content = {"subwords": input_subwords, "cluster": cluster_data[start_index:]}
    df = pd.DataFrame(data = content)
    return df

def back_to_string(words_list):
    """
    Recombines words from array into single string

    :param words_list: list of subwords
    """
    description = ""
    for word in words_list:
        description += (word + " ")
    return description

def extractTerms(k, df):
    """
    returns the top k terms of extracting query terms

    :param k the number of terms to get
    :param df the dataframe of subwords and cluster values
    return list of tuples (term, cluster value)
    """
    terms = []
    only_words = df["subwords"].to_list()

    description = back_to_string(only_words)

    vectorizer = TfidfVectorizer(use_idf= True)
    response = vectorizer.fit_transform([description])
    feature_names = vectorizer.get_feature_names_out()

    #get tfidf scores for input document
    tfidf_scores = {}
    for col in response.nonzero()[1]:
        tfidf_scores[feature_names[col]] = response[0, col]

    sorted_by_scores = sorted(tfidf_scores.items(), key=lambda x:x[1], reverse=True)
    for i in range(k):
        if i < len(sorted_by_scores):
            term = sorted_by_scores[i]
            og_word = ""
            for word in only_words:
                if "##" in word and term[0] in word:
                    og_word = word
                    break
                elif term[0] == word:
                    og_word = word
                    break
            row = df.loc[(df["subwords"] == og_word)]

            terms.append((term[0], row.iloc[0]["cluster"]))

    return terms
