"""
This script compares the training performance between my own implementation and sklearn MLPClassifier
"""

import numpy as np
from time import time

from sklearn.neural_network import MLPClassifier

from libNN.network import Network
from libNN.activation import ReLU, Sigmoid, Softmax
from libNN.cost import QuadraticCost, CrossEntropyCost
from libNN.regularizer import L2Regularizer, L1Regularizer
from libNN.optimizer import SGD, SGDMomentum, Adagrad, Adadelta, RMSprop, Adam
from libNN.utils import BatchNormalization as BN

# load data sets
train_X = np.load('../data/train_X.npy')
train_y = np.load('../data/train_y.npy')
val_X = np.load('../data/val_X.npy')
val_y = np.load('../data/val_y.npy')
test_X = np.load('../data/test_X.npy')
test_y = np.load('../data/test_y.npy')

"""
print("libNN:")
start_time = time()

# model
model = Network(sizes = [train_X.shape[1], 100, 100, train_y.shape[1]], 
                activations = [Sigmoid(), Sigmoid(), Sigmoid()], 
                cost = CrossEntropyCost(),
                batch_normalizations = [None, None, None])



# training
model.fit(data = train_X, 
          labels = train_y, 
          batch_size = 30, 
          epochs = 30, 
          learning_rate = 0.1, 
          validation_data = None,
          regularizer = None,
          optimizer = SGD(),
          plot_error = False,
          dropout = None,
          early_stopping = None,
          eval_accuracy = True,
          verbose = 0)

print("libNN costs {0} seconds".format(time() - start_time))

"""


"""
sklearn NN library
"""
print("sklearn NN:")
start_time = time()

# model
model = MLPClassifier(hidden_layer_sizes=(100,100), 
                      activation='logistic', 
                      solver="sgd", 
                      batch_size=30, 
                      learning_rate_init=0.1, 
                      max_iter=30)



# training
model.fit(train_X, train_y)
print("sklearn NN costs {0} seconds".format(time() - start_time))