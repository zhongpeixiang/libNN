import numpy as np
import matplotlib.pyplot as plt
import json

from libNN.optimizer import SGD

class Network(object):
    def __init__(self, sizes, activations, cost, batch_normalizations=None):
        """
        sizes: a list of size of each layer, including input and output layer
        activations: a list of activation functions corresponding to each layer except for input layer
        batch_normalizations: a list of BatchNormalization objects
        cost: cost function
        """
        # check dimension matching
        if len(sizes) != len(activations) + 1:
            raise ValueError('number of layers except input layer and number of activations do not match')
        elif (batch_normalizations != None) and (len(activations) != len(batch_normalizations)):
            raise ValueError('number of layers except input layer and number of batch normalization layers do not match')
        
        self.length = len(sizes)
        self.sizes = sizes
        self.activation_functions = activations
        self.batch_normalizations = batch_normalizations
        self.cost = cost
        
        
        # initialize params
        self.initialize_params()
    
    def initialize_params(self):
        """
        initialize weights and bias for each layer to corresponding size
        initialize batch normalization layers
        """
        # inti weights and biases
        self.W = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.random.randn(y, 1) for y in self.sizes[1:]]
        
        # init batch normalization layers
        for i in range(len(self.batch_normalizations)):
            if self.batch_normalizations[i]:
                self.batch_normalizations[i].init_params(self.sizes[i+1])
                
        
    def backpropagate(self, data, labels):
        """
        data: batch data in numpy.ndarray
        labels: batch labels in numpy.ndarray
        forward propagate inputs through the network to produce outputs and backpropagate errors
        """
        zs = []
        
        # input data as activations in the first layer
        a = data
        activations = [a]
        
        # loop through all layers
        for W, b, activation_function, BN in zip(self.W, self.b, self.activation_functions, self.batch_normalizations):
            z = np.dot(W, a) + b
            
            # batch normalization
            if BN:
                # print("---")
                z = BN.transform(z)
            
            zs.append(z)
            a = activation_function.evaluate(z)
            activations.append(a)
            
            
        """
        backpropagate errors to previous layers 
        """
        batch_size = labels.shape[1]
        
        # deltas for output layer
        delta = self.cost.delta(zs[-1], labels, self.activation_functions[-1])
        
        # if there is batch normalization at output layer
        if self.batch_normalizations[-1]:
            # print("---")
            delta = self.batch_normalizations[-1].backprop(delta)
        
        # weight and bias updates
        delta_weights = [np.zeros(W.shape) for W in self.W]
        delta_biases = [np.zeros(b.shape) for b in self.b]
        
        # delta regularized costs for weights and biases
        delta_reg_weights = [np.zeros(W.shape) for W in self.W]
        delta_reg_biases = [np.zeros(b.shape) for b in self.b]
        
        # output layer gradients
        delta_weights[-1] = np.matmul(delta, activations[-2].T)/batch_size
        delta_biases[-1] = np.mean(delta, axis = 1, keepdims=True)
            
        
        # hidden layer weights and biases
        for i in range(2, self.length):
            delta = np.dot(self.W[-i + 1].T, delta) * self.activation_functions[-i].prime(zs[-i])
            
            # backprop for batch normalization
            if self.batch_normalizations[-i]:
                # print("---")
                delta = self.batch_normalizations[-i].backprop(delta)
            
            # average updates over mini-batch
            delta_weights[-i] = np.dot(delta, activations[-i - 1].T)/batch_size
            delta_biases[-i] = np.mean(delta, axis = 1, keepdims=True)
            
        # regularized delta weights
        if self.regularizer:
            delta_reg_weights = self.regularizer.weight_derivative(self.W)
            delta_reg_biases = self.regularizer.bias_derivative(self.b)
        
        # update delta weights to include regularisation delta weights
        delta_weights = [dW + self._lambda * dRW/self.n for dW, dRW in zip(delta_weights, delta_reg_weights)]
        delta_biases = [db + self._lambda * dRb/self.n for db, dRb in zip(delta_biases, delta_reg_biases)]
        
        
        """
        update weights and biases
        """
        # use optimizer to update params
        self.W = self.optimizer.update_W(self.W, delta_weights, self.learning_rate)
        self.b = self.optimizer.update_b(self.b, delta_biases, self.learning_rate)
        
    
    def fit(self, data, labels, 
            learning_rate = 0.1, 
            epochs = 10, 
            batch_size = 16, 
            evaluation_data=None, 
            regularizer=None, 
            _lambda=0, 
            optimizer=SGD(),
            plot_error = False):
        """
        data: training features, numpy ndarray of size n*F, where n is number of training data and F is number of features in each sample
        labels: training labels, numpy ndarray of size n*1, where n is number of training data
        learning_rate: learning rate for gradient descent, 0.01 is a good starting point for Adaptive gradients (Adagrad, Adadelta, Adam, RMSprops, etc.)
        epochs: number of epochs for training
        batch_size: number of training samples in one batch
        regularizer: regularizer for cost function
        _lambda: regularization parameter
        optimizer: optimization algorithm for gradient descent and learning
        """
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self._lambda = _lambda
        self.optimizer = optimizer
        self.optimizer.set_sizes(self.sizes)
        self.training_errors = []
        self.validation_errors = []
        
        n = len(data)
        self.n = n
        dim_data = data.shape[1]
        dim_labels = labels.shape[1]
        
        # update weight std
        self.W = self.W/np.sqrt(dim_data)
        
        # set learning rate for batch normalization
        for i in range(len(self.batch_normalizations)):
            if self.batch_normalizations[i]:
                self.batch_normalizations[i].set_lr(self.learning_rate)
        
        # concatenate to one array for shuffling
        data_labels = np.concatenate((data, labels), axis = 1)
        
        for e in range(epochs):
            
            # training accuracy for after each epoch
            print("Epoch ", e + 1)
            
            # randomly shuffle data and labels for each epoch
            np.random.shuffle(data_labels)
            data = data_labels[:,0:-dim_labels]
            labels = data_labels[:,-dim_labels:]
            
            # initialize sum of squares of gradients
            self.SS_delta_weights = [np.zeros(W.shape) for W in self.W]
            self.SS_delta_biases = [np.zeros(b.shape) for b in self.b]
            
            # get mini-batch data
            batch_indexes = [i for i in range(0, n, batch_size)]
            
            # reset timestep t
            self.optimizer.t = 0
            for i in batch_indexes:
                batch_data = data[i: i + batch_size]
                batch_labels = labels[i: i + batch_size]
                
                # forward and back propagate errors and update weights and biases
                self.backpropagate(batch_data.T, batch_labels.T)
            
            # training and evaluation accuracies
            training_error = self.score(data, labels)
            self.training_errors.append(training_error)
            print("Training accuracy: ", training_error)
            if evaluation_data:
                validation_error = self.score(evaluation_data[0], evaluation_data[1])
                self.validation_errors.append(validation_error)
                print("Evaluation accuracy: ", validation_error)
        
        # plot training and validation errors
        if plot_error:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.training_errors)
            ax.plot(self.validation_errors)
            fig.show()
        
    def score(self, data, labels):
        predictions = self.predict(data.T)
        true_labels = np.argmax(labels, axis = 1)
        return sum(predictions == true_labels)/len(labels)
    
    
    def predict(self, data):
        # feedforward
        a = data
        for W, b, activation_function, BN in zip(self.W, self.b, self.activation_functions, self.batch_normalizations):
            z = np.dot(W, a) + b
            
            # batch normalization for prediction
            if BN:
                z = BN.prediction_transform(z)
            
            a = activation_function.evaluate(z)
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
    
    
   
        