import numpy as np
from time import time

from utils.treebank import StanfordSentiment
from libNN.network import Word2Vec


# load data sets
dataset = StanfordSentiment()

# model
model = Word2Vec(word_dim = 50)

# training
start_time = time()
word_vectors = model.fit(dataset = dataset)

# save
print("Saving word vectors...")
np.save("word_vectors", word_vectors)

print("Training took {0} seconds".format(time() - start_time))
print(word_vectors[0:5])

