import numpy as np
from tensorflow.keras import *


class Emojify:
    def __init__(self, input_shape, words_to_vec_map, words_to_index):
        """
        :param input_shape: shape of the input
        :param words_to_vec_map: dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
        :param words_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words)
        """

        self.input_shape = input_shape
        self.words_to_vec_map = words_to_vec_map
        self.words_to_index = words_to_index

    def pretrained_embedding_layer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
        :return: pretrained layer Keras instance
        """

        vocab_len = len(self.words_to_index) + 1
        emb_dim = self.words_to_vec_map['chechen'].shape[0]       # (=50)

        # Initialize the embedding matrix as a numpy array of zeros.
        emb_matrix = np.zeros((vocab_len, emb_dim))

        # Set each row "idx" of the embedding matrix to be
        # the word vector representation of the idx'th word of the vocabulary
        for word, idx in self.words_to_index.items():
            emb_matrix[idx, :] = self.words_to_vec_map[word]

        # Define Keras embedding layer. Make it non-trainable.
        embedding_layer = layers.Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)

        embedding_layer.build((None, ))
        embedding_layer.set_weights([emb_matrix])

        return embedding_layer

    def build(self):
        """
        Function creating the Emojify-v2 model's graph.
        :return: a model instance in Keras
        """

        # Define sentence_indices as the input of the graph.
        # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
        sentence_indices = layers.Input(shape=self.input_shape, dtype='int32')
        # Create the embedding layer pretrained with GloVe Vectors

        embedding_layer = Emojify.pretrained_embedding_layer(self)

        # Propagate sentence_indices through your embedding layer
        embeddings = embedding_layer(sentence_indices)

        # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
        X = layers.LSTM(128, return_sequences=True)(embeddings)
        # Add dropout with a probability of 0.5
        X = layers.Dropout(0.5)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden state
        X = layers.LSTM(128, return_sequences=False)(X)
        X = layers.Dropout(0.5)(X)
        # Propagate X through a Dense layer with 5 units and add softmax activation
        X = layers.Dense(5, activation="softmax")(X)

        model = models.Model(inputs=sentence_indices, outputs=X)

        return model
