import numpy as np
from time import time

from libNN.network import GloVe


# model
model = GloVe(word_dim = 200)

# corpus directory
start_time = time()
corpus = "/media/external/peixiang/NLP/data/wiki-corpus/latest/"
data_dir = "/media/external/peixiang/NLP/data/trained/"

# training
"""
model.fit(corpus, 
          window_size = 8,
          batch_size = 10000,
          load_vocab=data_dir + "vocab.csv", 
          save_matrix=data_dir + "matrix.npy", 
          save_word_vectors=data_dir + "word_vectors.npy")
"""

# epochs _ word_dim _ batch_size for save_word_vectors file name format
model.fit(corpus, 
          window_size = 8,
          batch_size = 1000,
          load_vocab=data_dir + "vocab.csv", 
          load_matrix=data_dir + "matrix.npy", 
          save_word_vectors=data_dir + "word_vectors_10_200_1000.npy")


print("Training took {0} seconds".format(time() - start_time))


