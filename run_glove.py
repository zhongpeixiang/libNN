import numpy as np
from time import time

from libNN.network import GloVe


# model
model = GloVe(word_dim = 200)

# corpus directory
start_time = time()
corpus = "/media/foodAIDisk/peixiang/NLP/data/wiki-corpus/latest/"
data_dir = "/media/foodAIDisk/peixiang/NLP/data/trained/"

# training
"""
model.fit(corpus, 
          window_size = 8,
          save_vocab=data_dir + "vocab.csv", 
          save_matrix=data_dir + "matrix.npy", 
          save_word_table=data_dir + "word_table.csv",
          save_word_vectors=data_dir + "word_vectors.npy")
"""

model.fit(corpus, 
          window_size = 8,
          batch_size = 10000,
          load_vocab=data_dir + "vocab.csv", 
          load_matrix=data_dir + "matrix.npy", 
          save_word_vectors=data_dir + "word_vectors.npy")

print("Training took {0} seconds".format(time() - start_time))


