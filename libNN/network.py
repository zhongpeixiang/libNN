import numpy as np
import matplotlib.pyplot as plt
import json
import random

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
        # get dropout masks for this mini-batch
        dropout_masks = self.get_dropout_masks()
        
        zs = []
        
        # input data as activations in the first layer
        a = data
        
        # dropout for input layer activations and scale by 1/p
        a = a * dropout_masks[0]/self.dropout_p[0]
        
        activations = [a]
        
        ###############
        # feedforward #
        ###############
        # loop through all layers
        for W, b, activation_function, BN, dropout_mask, p in zip(self.W, self.b, self.activation_functions, self.batch_normalizations, dropout_masks[1:], self.dropout_p[1:]):
            
            # feedforward
            z = np.dot(W, a) + b
            
            # batch normalization
            if BN:
                # print("---")
                z = BN.transform(z)
            
            zs.append(z)
            a = activation_function.evaluate(z)
            
            # drop out some units and scale active units by 1/p, where p  = 1 - dropout
            a = a * dropout_mask/p
            
            activations.append(a)
            
            
        """
        backpropagate errors to previous layers 
        """
        ############
        # backprop #
        ############
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
            
            # dropout
            delta = delta * dropout_masks[-i]
            
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
        #################
        # params update #
        #################
        # use optimizer to update params
        self.W = self.optimizer.update_W(self.W, delta_weights, self.learning_rate)
        self.b = self.optimizer.update_b(self.b, delta_biases, self.learning_rate)
        
    
    def fit(self, data, labels, 
            learning_rate = 0.1, 
            epochs = 10, 
            batch_size = 16, 
            validation_data=None, 
            regularizer=None, 
            optimizer=SGD(),
            plot_error = False,
            dropout = None,
            early_stopping = (5, 1),
            eval_accuracy = False,
            verbose = 0):
        """
        data: training features, numpy ndarray of size n*F, where n is number of training data and F is number of features in each sample
        labels: training labels, numpy ndarray of size n*1, where n is number of training data
        learning_rate: learning rate for gradient descent, 0.01 is a good starting point for Adaptive gradients (Adagrad, Adadelta, Adam, RMSprops, etc.)
        epochs: number of epochs for training
        batch_size: number of training samples in one batch
        regularizer: regularizer for cost function
        optimizer: optimization algorithm for gradient descent and learning
        plot_error: set True to plot training and validation error against epochs
        dropout: a list of dropout strengths for input and hidden layers, 0 means no dropout and 1 means dropping out all units
        early_stopping: a tuple (number of consecutive epochs tested, threshold) to implement early stopping to reduce overfitting, set to False if no early_stopping
        verbose: show training errors and accuracies if not 0
        """
        if dropout != None and len(dropout) != self.length - 1:
            raise ValueError('number of dropout strengths does not match with number of layers except output layer')
            
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self._lambda = 0
        if self.regularizer:
            self._lambda = self.regularizer._lambda
        self.optimizer = optimizer
        self.optimizer.set_sizes(self.sizes)
        self.training_errors = []
        self.training_accuracies = []
        self.validation_errors = []
        self.validation_accuracies = []
        self.early_stopping = early_stopping
        self.eval_accuracy = eval_accuracy
        self.verbose = verbose
        
        ###########
        # dropout #
        ###########
        if dropout:
            self.dropout_p = [1 - _ for _ in dropout] # probability of retaining a unit
            self.dropout_p.append(1) # retaining rate for output layer is always 1
        else:
            self.dropout_p = [1] * self.length # no dropout, all retaining rates are 1        
        
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
        
        ############
        # training #
        ############
        
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
                
                ############
                # backprop #
                ############
                # forward and back propagate errors and update weights and biases
                self.backpropagate(batch_data.T, batch_labels.T)
            
            
            ##############
            # monitoring #
            ##############
            if self.verbose != 0:
                # training errors and accuracies
                training_error = self.error(data, labels)
                self.training_errors.append(training_error)
                print("Training error: ", training_error)
                
                if self.eval_accuracy:
                    training_accuracy = self.accuracy(data, labels)
                    self.training_accuracies.append(training_accuracy)
                    print("Training accuracy: ", training_accuracy)
            
            # validation errors and accuracies
            if validation_data:
                validation_error = self.error(validation_data[0], validation_data[1])
                self.validation_errors.append(validation_error)
                print("Validation error: ", validation_error)
                
                if self.eval_accuracy:
                    validation_accuracy = self.accuracy(validation_data[0], validation_data[1])
                    self.validation_accuracies.append(validation_accuracy)
                    print("Evaluation accuracy: ", validation_accuracy)
                
                # early stopping
                if self.early_stopping != False and self.early_stopping != None:
                    min_error = max(self.validation_errors)
                    """
                    if validation errors of last a few consecutive epochs are all larger than 
                    a min validation error so far by a certain threshold, then stop training. 
                    """
                    if all((100*(np.array(self.validation_errors[-self.early_stopping[0]:])/min_error - 1)) > self.early_stopping[1]):
                        print("Early stopping!!!")
                        break
        
        ############
        # plotting #
        ############
        # plot training and validation accuracies
        if plot_error:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.training_errors)
            ax.plot(self.validation_errors)
            fig.show()
        
    
    def get_dropout_masks(self):
        """
        get dropout masks for all layers except output, returns a list of arrays of 0s and 1s
        """
        return [(np.random.rand(size, 1) > (1 - p))*1 for size, p in zip(self.sizes, self.dropout_p)]
    
    
    
    def error(self, data, labels):
        # return cost function evaluation + regularization cost
        # cost evaluated by cost function
        predictions = self.predict(data.T)
        cost = self.cost.evaluate(predictions, labels.T)
        
        # add regularization cost
        if self.regularizer:
            cost += self.regularizer.evaluate(self.W)
        cost = cost/len(data)
        return cost
    
    
    def accuracy(self, data, labels):
        # accuracy: true predictions / total predictions
        predictions = self.predict(data.T)
        predictions = np.argmax(predictions, axis = 0)
        true_labels = np.argmax(labels, axis = 1)
        return sum(predictions == true_labels)/len(labels)
    
    
    def predict(self, data):
        # feedforward and return output activations
        a = data
        for W, b, activation_function, BN in zip(self.W, self.b, self.activation_functions, self.batch_normalizations):
            z = np.dot(W, a) + b
            
            # batch normalization for prediction
            if BN:
                z = BN.prediction_transform(z)
            
            a = activation_function.evaluate(z)
        return a
    
    
    
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
    
    
    
    
class Word2Vec(object):
    def __init__(self, word_dim):
        self.word_dim = word_dim
        
    
    def init_weights(self):
        # initialize W1 to uniform distribution over [-0.5, 0.5), initialize W2 to all zeros
        self.W1 = np.random.rand(self.word_dim, self.num_words) - 0.5 # (word_dim, num_words)
        self.W2 = np.zeros((self.num_words, self.word_dim)) # (num_words, word_dim)
    
    
    def fit(self, dataset, model = "skipgram", C = 10, K = 10, learning_rate = 0.1, batch_size = 50, epochs = 1):
        """
        model: string, "skipgram" or "cbow"
        dataset: the corpus to train
        C: window size of context
        K: number of negative samples when using negative sampling
        learning_rate: learning rate for gradient descent
        batch_size: batch size for sgd
        epochs: number of epochs to train
        """
        self.model = model
        self.dataset = dataset
        self.C = C
        self.K = K
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # get tokens and word count
        self.tokens = self.get_tokens()
        self.word_count = self.word_count() # total word count
        # print("total word count: ", self.word_count)
        # print("distinct word count: ", len(self.tokens))
        self.niters = int(self.word_count/self.batch_size) # number of iterations
        
        # initialize weights
        self.num_words = len(self.tokens) # unique tokens
        self.init_weights()
        
        # train
        for e in range(self.epochs):
            
            
            # feedforward
            for i in range(self.niters):
                if i % 100 == 0:
                    print("running iteration {0}, progress: {1}%".format(i, round(100*i/self.niters, 2)))
                # get a batch of random context of size K in dataset
                batch_contexts = [self.get_context() for j in range(batch_size)]
                cost, delta_W1, delta_W2 = self.word2vec_sgd(batch_contexts)
                
                # weights update
                self.W1 -= self.learning_rate * delta_W1
                self.W2 -= self.learning_rate * delta_W2
        
        # after training
        return (self.W1.T + self.W2)
    
    def word2vec_sgd(self, batch_contexts):
        """
        batch stochastic gradient descent
        """
        cost = 0
        delta_W1 = np.zeros(self.W1.shape)
        delta_W2 = np.zeros(self.W2.shape)
        for context in batch_contexts:
            current_word = context[0]
            context = context[1]
            
            if self.model == "skipgram":
                _cost, _delta_W1, _delta_W2 = self.skipgram(current_word, context)
            if self.model == "cbow":
                _cost, _delta_W1, _delta_W2 = self.cbow(current_word, context)
            cost += _cost
            delta_W1 += _delta_W1
            delta_W2 += _delta_W2
        return cost/self.batch_size, delta_W1/self.batch_size, delta_W2/self.batch_size
    
    
    def skipgram(self, current_word, context):
        """
        implement skipgram model
        returns a tuple of cost, matrix 1 gradients and matrix 2 gradients
        """
        cost = 0
        delta_W1 = np.zeros(self.W1.shape)
        delta_W2 = np.zeros(self.W2.shape)
        for context_word in context:
            _cost, _delta_W1, _delta_W2 = self.negative_sampling_cost_gradients(current_word, context_word)
            cost += _cost
            delta_W1[:, self.tokens[current_word]] += _delta_W1
            delta_W2 += _delta_W2
        return cost, delta_W1, delta_W2
    
    
    def cbow(self, current_word, context):
        """
        implement cbow model
        returns a tuple of cost, matrix 1 gradients and matrix 2 gradients
        """
        cost = 0
        delta_W1 = np.zeros(self.W1.shape)
        delta_W2 = np.zeros(self.W2.shape)
        for context_word in context:
            _cost, _delta_W1, _delta_W2 = self.negative_sampling_cost_gradients(context_word, current_word)
            cost += _cost
            delta_W1[:, self.tokens[context_word]] += _delta_W1
            delta_W2 += _delta_W2
        return cost, delta_W1, delta_W2
    
    
    
    def negative_sampling_cost_gradients(self, current_word, context_word):
        deltaW2 = np.zeros(self.W2.shape) # (num_words, word_dim) 
        
        # get negative samples
        indices = [self.tokens[context_word]]
        indices.extend(self.get_negative_samples(context_word))
        
        output_words = self.W2[indices] # (K+1, word_dim)
        directions = np.array([1] + [-1] * self.K) # (K+1, 0)
        
        # feedforward
        z1 = self.W1[:,self.tokens[current_word]] # (word_dim, 0)
        z2 = np.dot(self.W2, z1) # (num_words, word_dim) dot (word_dim, 0) = (num_words, 0) 
        a2 = self.softmax(z2) # (word_dim, 0)
        
        # cost
        delta = self.sigmoid(directions * np.dot(output_words, z1)) # (K+1, 0)
        delta_minus1 = directions*(delta - 1) # (K+1, 0)
        cost = -np.sum(np.log(delta))
        
        # grad
        deltaW1 = np.dot(output_words.T, delta_minus1) # (word_dim, 0)
        _deltaW2 = np.dot(delta_minus1.reshape(self.K+1, 1), z1.reshape(1, self.word_dim)) # (K+1, word_dim) 
        
        for k in range(self.K+1):
            deltaW2[indices[k]] += _deltaW2[k]
        
        return cost, deltaW1, deltaW2
    
    def get_negative_samples(self, context_word):
        """
        context_word: a word string
        return a list of indices containing the context word index and K indices of negative samples
        """
        target = self.tokens[context_word]
        indices = []
        while len(indices) != self.K:
            idx = self.dataset.sampleTokenIdx()
            if idx != target:
                indices.append(idx)
        return indices
    
    
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    
    
    def softmax(self, z):
        z = z - np.max(z, axis = 0)
        return np.exp(z)/np.sum(np.exp(z), axis = 0)
    
    
    
    def get_tokens(self):
        """
        returns a dictionary mapping words into indices
        """
        tokens = self.dataset.tokens()
        return tokens 
        
    
    def get_context(self):
        """
        returns a tuple of center word and a list of context words
        """
        # the actual window size is no larger than K
        C1 = random.randint(1, self.C)
        center_word, context = self.dataset.getRandomContext(C1)
        return center_word, context
        
    
    def word_count(self):
        """
        returns total word count in dataset
        """
        word_count = self.dataset._wordcount
        return word_count