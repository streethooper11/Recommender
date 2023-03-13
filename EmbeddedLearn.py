# Source for using BERT:
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
# https://techblog.assignar.com/how-to-use-bert-sentence-embedding-for-clustering-text/

import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import re


def getTokenEmbeddings(trained_model, indexed_tokens, segments_ids):
    """
    Obtains token embeddings from the indexed tokens and segment IDs

    :param trained_model: Trained BERT model
    :param indexed_tokens: Indexed tokens
    :param segments_ids: List of segment IDs
    :return:The embeddings in the sentence
    """

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = trained_model(tokens_tensor, segments_tensors)

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


def indexAndSetSegmentIds(tokenizer_name: str, sentence: str):
    """
    Maps the token strings to vocabulary indices
    and mark the tokens

    :param tokenizer_name: The name of the tokenizer
    :param sentence: A given sentence
    :return: Indexed tokens and segment IDs
    """

    # Load pre-trained model tokenizer with a given name
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indices.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the tokens as belonging to sentence "1", as each element is 1 sentence.
    segments_ids = [1] * len(tokenized_text)

    return tokenized_text, indexed_tokens, segments_ids


def embedSentence(trained_model, tokenizer_name: str, sentence: str):
    """
    Performs word embedding with the given BERT model, tokenizer name,
    and a sentence.

    :param trained_model: The trained BERT model
    :param tokenizer_name: The name of the tokenizer
    :param sentence: Given sentence to create a vector for
    :return: A list of vectors that represents the words in the sentence
    """

    tokenized_text, indexed_tokens, segments_ids = indexAndSetSegmentIds(tokenizer_name, sentence)
    token_embeddings = getTokenEmbeddings(trained_model, indexed_tokens, segments_ids)

    # Stores the token vectors
    token_vecs_cat = []

    # For each token in the sentence...
    for token in token_embeddings:
        # Concatenate the vectors from the last four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    return tokenized_text, token_vecs_cat


def embedWords(csvLoc: str, bert_version: str):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(bert_version,
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    df = pd.read_csv(csvLoc)
    df_length = len(df.index)

    all_actors = []
    all_subwords = []
    all_vectors = []

    # used to remove separators and some special characters that appear in descriptions
    remove_words = set(["[CLS]", "[SEP]", ",", '"', "'", ";", ":", "!", "$", "^"])
    for i in range(df_length):
        paragraph = str(df.iloc[i, 1])
        sentences = paragraph.split(".")
        actor_paragraph = str(df.iloc[i, 0])

        word_token_paragraph = []
        vector_paragraph = []
        for sentence in sentences:
            word_token_sentence, vector_sentence = embedSentence(model, bert_version, re.sub(r"\[.*\]", "", sentence))
            word_token_paragraph.extend(word_token_sentence)  # Add a list of words from the sentence
            vector_paragraph.extend(vector_sentence)  # Get BERT tensors from the sentence

        # remove tensors for separators, some special characters, and 1-length characters
        info_paragraph = [t for t in zip(word_token_paragraph, vector_paragraph)
                          if (t[0] not in remove_words) and (len(t[0]) > 1)]
        # then split into lists again
        word_token_paragraph, vector_paragraph = zip(*info_paragraph)

        for j in range(len(word_token_paragraph)):
            all_actors.append(actor_paragraph)

        all_subwords.extend(word_token_paragraph)  # Add a list of words from the paragraph
        all_vectors.extend(vector_paragraph)  # Get BERT tensors from the paragraph

    return all_actors, all_subwords, all_vectors
