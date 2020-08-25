#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
from cnn import CNN
from highway import Highway

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        pad_token_idx = vocab['<pad>']
        self.embeddings = nn.Embedding(len(vocab), embed_size, padding_idx=pad_token_idx)
        ## End A4 code
        self.cnn = CNN(embed_size, num_filters=embed_size, kernel_size=5)
        ### YOUR CODE HERE for part 1j
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        embedded = self.embeddings(input)
        # embedded is (sentence_length, batch_size, max_word_length, embed_size)
        embedded = embedded.permute([0, 1, 3, 2])
        # embedded is (sentence_length, batch_size, embed_size, max_word_length)

        reshaped = embedded.reshape(-1, embedded.shape[-2], embedded.shape[-1])
        # reshaped is (sentence_length * batch_size, embed_size, max_word_length)
        conv = self.cnn(reshaped)
        # conv is (sentence_length * batch_size, embed_size)
        #highway_inp =

        highway_out = self.highway(conv)

        # highway_out (sentence_length * batch_size, embed_size)

        output = highway_out.reshape(input.shape[0], input.shape[1], conv.shape[-1])
        # output is (sentence_length, batch_size, embed_size)

        return self.dropout(output)

        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j


        ### END YOUR CODE
