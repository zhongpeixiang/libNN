import numpy as np

from libNN.utils import load

# load data sets
train_X = np.load('data/train_X.npy')
train_y = np.load('data/train_y.npy')
val_X = np.load('data/val_X.npy')
val_y = np.load('data/val_y.npy')
test_X = np.load('data/test_X.npy')
test_y = np.load('data/test_y.npy')

# load model
model = load("saved_models/model_1502417142")

# before further training
score = model.score(test_X, test_y)
print("------------------------------------")
print("Original Accuracy: ", score)


# training
model.fit(train_X[-10000:], train_y[-10000:], batch_size = 32, epochs = 10, learning_rate = 0.5, evaluation_data = (val_X, val_y))

# after training
score = model.score(test_X, test_y)
print("------------------------------------")
print("New Accuracy: ", score)
