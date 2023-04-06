#!/usr/bin/env python3
"""
Responsible for generating actor information with cluster information,
"""

from transformers import BertModel, BertTokenizer


def setupBert():
    # SETUP
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Load pre-trained model tokenizer with a given bert version
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer