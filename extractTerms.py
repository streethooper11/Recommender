#https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
#https://stackoverflow.com/questions/60642043/how-fit-transform-transform-and-tfidfvectorizer-works

import pandas as pd
import numpy as np
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

def extractTerms(k, df):
    """
    returns the top k terms of extracting query terms

    :param k the number of terms to get
    :param df the dataframe of subwords and cluster values
    """
    terms = []
    only_words = df["subwords"].to_list()

    vectorizer = TfidfVectorizer(use_idf= True)
    response = vectorizer.fit_transform(only_words)
    feature_names = vectorizer.get_feature_names_out()

    return terms
    
