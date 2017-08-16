import numpy as np
from time import time

from libNN.network import Network
from libNN.activation import ReLU, Sigmoid, Softmax
from libNN.cost import QuadraticCost, CrossEntropyCost
from libNN.regularizer import L2Regularizer, L1Regularizer
from libNN.optimizer import SGD, Adagrad, Adadelta, RMSprop, Adam
from libNN.utils import BatchNormalization as BN

# load data sets
train_X = np.load('../data/train_X.npy')
train_y = np.load('../data/train_y.npy')
val_X = np.load('../data/val_X.npy')
val_y = np.load('../data/val_y.npy')
test_X = np.load('../data/test_X.npy')
test_y = np.load('../data/test_y.npy')

# model
model = Network(sizes = [train_X.shape[1], 30, train_y.shape[1]], 
                activations = [Sigmoid(), Sigmoid()], 
                cost = CrossEntropyCost(),
                batch_normalizations = [BN(), BN()])

# training
model.fit(data = train_X[:10000], 
          labels = train_y[:10000], 
          batch_size = 20, 
          epochs = 10, 
          learning_rate = 0.1, 
          evaluation_data = (val_X, val_y),
          regularizer = L2Regularizer(),
          _lambda = 0,
          optimizer = SGD(),
          plot_error = False)

# score
score = model.score(test_X, test_y)
print("------------------------------------")
print("Testing Accuracy: ", score)


# save model
# filename = "saved_models/" + "model_" + str(int(time()))
# model.save(filename)