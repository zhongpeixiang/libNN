import numpy as np
from time import time

from libNN.network import GloVe


# model
model = GloVe(word_dim = 50)

# corpus directory
start_time = time()
corpus = "/media/foodAIDisk/peixiang/NLP/data/wiki-corpus-sample/"
save_dir = "/media/foodAIDisk/peixiang/NLP/data/trained/"

# training
model.fit(corpus, save_vocab=save_dir + "vocab.csv", save_matrix=save_dir + "matrix.npy")

print("Training took {0} seconds".format(time() - start_time))


