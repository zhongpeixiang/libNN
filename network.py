import numpy as np
import json

class Network(object):
    def __init__(self, sizes, activations, cost):
        """
        sizes: a list of size of each layer, including input and output layer
        activations: a list of activation functions corresponding to each layer except for input layer
        cost: cost function
        """
        # check dimension matching
        if len(sizes) != len(activations) + 1:
            raise ValueError('number of layers except input layer and number of activations do not match')
        
        self.length = len(sizes)
        self.sizes = sizes
        self.activation_functions = activations
        self.cost = cost
        
        
        # initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        initialize weights and bias for each layer to corresponding size
        """
        self.W = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.random.randn(y, 1) for y in self.sizes[1:]]
    
        
        
    def backpropagate(self, data, labels):
        """
        data: batch data in numpy.ndarray
        labels: batch labels in numpy.ndarray
        forward propagate inputs through the network to produce outputs and backpropagate errors
        """
        zs = []
        activations = []
        
        # input data as activations in the first layer
        a = data
        activations.append(a)
        
        # loop through all layers
        for W, b, activation_function in zip(self.W, self.b, self.activation_functions):
            z = np.dot(W, a) + b
            zs.append(z)
            a = activation_function.evaluate(z)
            activations.append(a)
            
            
        """
        backpropagate errors to previous layers and update weights and biases
        """
        batch_size = labels.shape[1]
        
        # deltas for output layer
        delta = self.cost.delta(zs[-1], labels, self.activation_functions[-1])
        
        # weight and bias updates
        delta_weights = [np.zeros(W.shape) for W in self.W]
        delta_biases = [np.zeros(b.shape) for b in self.b]
        
        # delta regularized costs for weights and biases
        delta_reg_weights = [np.zeros(W.shape) for W in self.W]
        delta_reg_biases = [np.zeros(b.shape) for b in self.b]
        
        # output layer gradients
        delta_weights[-1] = np.matmul(delta, activations[-2].T)/batch_size
        delta_biases[-1] = np.mean(delta, axis = 1, keepdims=True)
        

        for i in range(2, self.length):
            delta = np.dot(self.W[-i + 1].T, delta) * self.activation_functions[-i].prime(zs[-i])
            
            # average updates over mini-batch
            delta_weights[-i] = np.dot(delta, activations[-i - 1].T)/batch_size
            delta_biases[-i] = np.mean(delta, axis = 1, keepdims=True)
            
        # regularized delta weights
        if self.regularizer:
            delta_reg_weights = self.regularizer.weight_derivative(self.W)
            delta_reg_biases = self.regularizer.bias_derivative(self.b)
        
        
        # update weights and biases
        self.W = [W - self.learning_rate * (dW + self._lambda*dRW/self.n) 
                  for W, dW, dRW in zip(self.W, delta_weights, delta_reg_weights)]
        self.b = [b - self.learning_rate * (db + self._lambda*dRb/self.n) 
                  for b, db, dRb in zip(self.b, delta_biases, delta_reg_biases)]
    
    def fit(self, data, labels, learning_rate = 0.1, epochs = 10, batch_size = 16, evaluation_data=None, regularizer=None, _lambda=0):
        """
        data: training features, numpy ndarray of size n*F, where n is number of training data and F is number of features in each sample
        labels: training labels, numpy ndarray of size n*1, where n is number of training data
        learning_rate: learning rate for gradient descent
        epochs: number of epochs for training
        batch_size: number of training samples in one batch
        regularizer: regularizer for cost function
        _lambda: regularization parameter
        """
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self._lambda = _lambda
        
        n = len(data)
        self.n = n
        dim_data = data.shape[1]
        dim_labels = labels.shape[1]
        
        # update weight std
        self.W = self.W/np.sqrt(dim_data)
        
        # concatenate to one array for shuffling
        data_labels = np.concatenate((data, labels), axis = 1)
        
        for e in range(epochs):
            
            # training accuracy for after each epoch
            print("Epoch ", e + 1)
            
            # randomly shuffle data and labels for each epoch
            np.random.shuffle(data_labels)
            data = data_labels[:,0:-dim_labels]
            labels = data_labels[:,-dim_labels:]
            
            # get mini-batch data
            batch_indexes = [i for i in range(0, n, batch_size)]
            for i in batch_indexes:
                batch_data = data[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                
                # forward and back propagate errors and update weights and biases
                self.backpropagate(batch_data.T, batch_labels.T)
            
            # training and evaluation accuracies
            print("Training accuracy: ", self.score(data, labels))
            if evaluation_data:
                print("Evaluation accuracy: ", self.score(evaluation_data[0], evaluation_data[1]))
        
    def score(self, data, labels):
        predictions = self.predict(data.T)
        true_labels = np.argmax(labels, axis = 1)
        return sum(predictions == true_labels)/len(labels)
    
    
    def predict(self, data):
        # feedforward
        a = data
        for W, b, activation_function in zip(self.W, self.b, self.activation_functions):
            a = activation_function.evaluate(np.dot(W, a) + b)
        return np.argmax(a, axis = 0)
    
    # save model into json file
    def save(self, filename):
        print("Saving model into " + filename + " ...")
        data = {
            "sizes": self.sizes,
            "activations": [str(a.__class__.__name__) for a in self.activation_functions],
            "W": [W.tolist() for W in self.W],
            "b": [b.tolist() for b in self.b],
            "cost": str(self.cost.__class__.__name__)
        }
        
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        print("Saving completed")
    
    
   
        