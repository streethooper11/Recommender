#!/usr/bin/env python3
"""
File responsible for word embedding using BERT
"""
import json
import string

# Source for using BERT:
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
# https://is-rajapaksha.medium.com/bert-word-embeddings-deep-dive-32f6214f02bf
# https://huggingface.co/docs/transformers/main_classes/tokenizer

import torch
import pandas as pd
import re

from Logic import preprocess, processList


def getTokenEmbeddings(trained_model, indexed_tokens):
    """
    Obtains token embeddings from the indexed tokens and segment IDs

    :param trained_model: Trained BERT model
    :param indexed_tokens: Indexed tokens
    :return:The embeddings in the sentence
    """

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    # Tokens were manually added with no pads; everything should be paid attention to, which is value of 1
    attention_masks = torch.tensor([[1] * len(indexed_tokens)])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = trained_model(tokens_tensor, attention_masks)

        # Evaluating the model will return a different number of objects based on
        # how it's configured in the trained model.
        # As `output_hidden_states = True` in the trained model, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    return token_embeddings


def tokenizeSentence(tokenizer, sentence: str):
    """
    Maps the token strings to vocabulary indices
    and mark the tokens

    :param tokenizer: BERT tokenizer
    :param sentence: A given sentence
    :return: Text with tokens and their indices
    """

    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indices.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    return tokenized_text, indexed_tokens


def embedSentence(trained_model, tokenizer, sentence: str):
    """
    Performs word embedding with the given BERT model, tokenizer name,
    and a sentence.

    :param trained_model: The trained BERT model
    :param tokenizer: BERT tokenizer
    :param sentence: Given sentence to create a vector for
    :return: A list of vectors that represents the words in the sentence
    """

    tokenized_text, indexed_tokens = tokenizeSentence(tokenizer, sentence)
    token_embeddings = getTokenEmbeddings(trained_model, indexed_tokens)

    # The second last hidden layer is used for efficiency yet high F1 score
    # Source: https://is-rajapaksha.medium.com/bert-word-embeddings-deep-dive-32f6214f02bf

    # Stores the token vectors, with shape [22 x 768]
    token_vecs_cat = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.
    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # append the last hidden layer, so each vector will be 768
        token_vecs_cat.append(token[-2])

    return tokenized_text, token_vecs_cat


def embedRoleDescription(model, tokenizer, paragraph: str):
    sentences = paragraph.split(".")  # split paragraph into sentences
    word_token_paragraph = []
    vector_paragraph = []

    for sentence in sentences:
        # Work on each sentence; remove [*] to remove references such as [1]
        word_token_sentence, vector_sentence = embedSentence(model, tokenizer, re.sub(r"\[.*]", "", sentence))
        word_token_paragraph.extend(word_token_sentence)  # Add a list of sub words from the sentence
        vector_paragraph.extend(vector_sentence)  # Get BERT tensors from the sentence

    return word_token_paragraph, vector_paragraph


def embedWords(csvLoc: str, model, tokenizer):
    # Word embedding each sentence could be done easily using flairNLP \
    # (https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/TRANSFORMER_EMBEDDINGS.md)
    # and their encode_plus library.
    # However, it does not provide the tokens; aka the subwords that BERT creates in the middle.
    # We preprocess to remove stop words, special tokens and punctuations before clustering,
    # which requires comparing stopwords with the subwords in the same index with actor names and embeddings
    # therefore, we will process the word embedding manually, such that we keep track of the subwords.

    df = pd.read_csv(csvLoc, header=None, encoding="utf-8", encoding_errors="replace")
    df_length = len(df.index)
    all_actors = []
    all_subwords = []
    all_vectors = []
    actor_counts = dict()

    # define stop words
    bert_tokens = ["[CLS]", "[SEP]"]
    with open("Data/NLTK_stopwords.txt", "r") as f:
        nltk = f.readlines()

    nltk_list = [line.strip() for line in nltk]

    stop_words = set(nltk_list + list(string.punctuation) + bert_tokens)

    for i in range(df_length):
        actor_name = str(df.iloc[i, 0])
        paragraph = str(df.iloc[i, 1])
        token_paragraph, vector_paragraph = embedRoleDescription(model, tokenizer, paragraph)
        token_paragraph, vector_paragraph = preprocess.eliminateStopWords(token_paragraph, vector_paragraph, stop_words)
        all_actors.append(actor_name)
        all_subwords.append(token_paragraph)
        all_vectors.append(vector_paragraph)

        # As the actor appeared in a role description increase count by the number of subwords
        if actor_name not in actor_counts:
            actor_counts[actor_name] = 0
        actor_counts[actor_name] += len(token_paragraph)

    return all_actors, all_subwords, all_vectors, actor_counts

