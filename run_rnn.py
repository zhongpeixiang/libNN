import numpy as np
from time import time

from libNN.network_gpu import RNN, DeepRNN


# model
model = DeepRNN(hidden_layer_sizes = [200, 200])

# training
start_time = time()
model.fit(corpus="/media/external/peixiang/NLP/data/wiki-corpus-sample/raw/", generate_text_length=200)

print("Training took {0} seconds".format(time() - start_time))

