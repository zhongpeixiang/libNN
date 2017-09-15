from os import listdir
from os.path import isfile, join
import operator
import csv

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
    
    
    
"""
Word2Vec Word Embedding
"""    
    
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
    
    
    

    
"""
GloVe Word Embedding
"""
class GloVe(object):
    def __init__(self, word_dim):
        self.word_dim = word_dim
    
    def vocab_list(self):
        """
        construct a vocabulary dictionary of unigram counts
        """
        print("Building vocabulary dictionary...")
        # return dictionary of word counts
        vocab = {}
        counter = 0
        
        # a list of filenames
        self.text_filenames = [join(self.corpus, f) for f in listdir(self.corpus) if isfile(join(self.corpus, f)) and not f.startswith('.')]
        self.num_files = len(self.text_filenames)
        print("List of corpus files: ")
        for filename in self.text_filenames:
            print(filename)
        
        
        if self.load_vocab:
            print("Loading vocabulary files... ")
            with open(self.load_vocab, 'r') as csv_file:
                reader = csv.reader(csv_file)
                vocab = {}
                for row in reader:
                    vocab[row[0]] = int(row[1])
            print("Loading vocabulary files completed. ")
        else:
            # loop through every file and get word counts
            for text_filename in self.text_filenames:
                counter += 1
                textfile = open(text_filename, "r")
                words = textfile.read().split(" ")

                for word in words:
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
                textfile.close()
                print("Building vocabulary progress: {0}%".format(100*counter/self.num_files))
        self.vocab = vocab
        
        # a list of tuples sorted by word counts, capped by vocab limit
        self.sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)[:self.vocab_limit]
        
        # a dictionary mapping word to its index in cooccurrence matrix
        self.word_table = {item[0]: index for index, item in enumerate(self.sorted_vocab)}
        
        # vocab size
        self.num_word = len(self.word_table)
        
        # save vocab dictionary
        if self.save_vocab:
            with open(self.save_vocab, 'w') as csv_file:
                print("Saving vocabulary dictionary...")
                writer = csv.writer(csv_file)
                for key, value in vocab.items():
                    writer.writerow([key, value])
                print("Saving vocabulary dictionary completed.")
        if self.save_word_table:
            with open(self.save_word_table, 'w') as csv_file:
                print("Saving word table...")
                writer = csv.writer(csv_file)
                for key, value in self.word_table.items():
                    writer.writerow([key, value])
                print("Saving word table completed.")
    
    def cooccurrence_matrix(self):
        """
        construct a cooccurrence matrix
        """
        # initialize matrix
        matrix = np.zeros((self.num_word, self.num_word))
        window_size = self.window_size
        word_table = self.word_table
        counter = 0
        
        if self.load_matrix:
            print("Loading cooccurrence matrix... ")
            matrix = np.load(self.load_matrix)
            print("Loading cooccurrence matrix completed. ")
        else:
            # loop through every file and get word counts
            for text_filename in self.text_filenames:
                counter += 1
                textfile = open(text_filename, "r")
                words = textfile.read().split(" ")
                words_size = len(words)

                for index, word in enumerate(words):
                    if word in word_table:
                        # context_words = words[max(index - window_size, 0) : index] + words[index + 1 : index + window_size + 1]
                        # context_indices = [word_table[context_word] for context_word in context_words if context_word in word_table]
                        # matrix[word_table[word], context_indices] += 1
                        
                        # weighted word occurrence count by distance
                        for context_index in range(max(index - window_size, 0), min(index + window_size + 1, words_size)):
                            if context_index != index and words[context_index] in word_table:
                                matrix[word_table[word], word_table[words[context_index]]] += 1/np.abs(context_index - index)
                textfile.close()
                print("Building matrix progress: {0}%".format(100*counter/self.num_files))
        
        # self.matrix = matrix
        
        # to reduce size of vocabulary for faster training
        self.matrix = matrix[:10000, :10000]
        self.num_word = 10000
        
        if self.save_matrix:
            print("Saving cooccurrence matrix...")
            np.save(self.save_matrix, matrix)
            print("Saving cooccurrence matrix completed")
    
    
    
    def train(self):
        """
        train word vectors
        """
        self.init_params()
        
        for t in range(self.epochs):
            print("Epoch {0} of {1}...".format(t + 1, self.epochs))
            # shuffle row and column index for minibatch sampling
            shuffled_row = np.random.permutation(self.num_word)
            shuffled_col = np.random.permutation(self.num_word)
            
            counter = 0

            for i in shuffled_row:
                counter += 1
                if counter % 1000 == 0:
                    print("Processed {0} samples...".format(counter))
                batch_indices = [i for i in range(0, self.num_word, self.batch_size)]
                for j in batch_indices:
                    dW1 = np.zeros(self.W1.shape)# (num_word, word_dim), W1 derivative matrix
                    dW2 = np.zeros(self.W2.shape)
                    
                    db1 = np.zeros(self.b1.shape) # (num_word, 1), bias derivative vector
                    db2 = np.zeros(self.b2.shape)
                    
                    js = shuffled_col[j : j + self.batch_size] # column indices for this mini batch
                    batch_data = self.matrix[i, js] # Xij values
                    weighting = np.minimum((batch_data/self.x_max)**self.alpha, 1) # weightings for Xij
                    
                    # gradients
                    dW1[i] = (1/self.batch_size) * 2 * np.dot(weighting * (np.dot(self.W1[i], self.W2[js].T) + self.b1[i] + self.b2[js] - np.log(1 + batch_data)), self.W2[js])
                    db1[i] =  np.mean(2 * weighting * (np.dot(self.W1[i], self.W2[js].T) + self.b1[i] + self.b2[js] - np.log(1 + batch_data)))
                    dW2[js] = (1/self.batch_size) * 2 * np.dot((weighting * (np.dot(self.W1[i], self.W2[js].T) + self.b1[i] + self.b2[js] - np.log(1 + batch_data))).reshape(self.batch_size, 1), self.W1[i].reshape(1, self.word_dim))
                    db2[js] = (1/self.batch_size) * 2 * weighting * (np.dot(self.W1[i], self.W2[js].T) + self.b1[i] + self.b2[js] - np.log(1 + batch_data))

                    # update params
                    self.W1 -= self.learning_rate * dW1
                    self.W2 -= self.learning_rate * dW2
                    self.b1 -= self.learning_rate * db1
                    self.b2 -= self.learning_rate * db2
                
                
        if self.save_word_vectors:
            # save word vectors
            print("Saving word vectors...")
            np.save(self.save_word_vectors, self.W1 + self.W2)
            print("Saving word vectors completed")
    
    
    def init_params(self):
        """
        initialize weight and bias parameters for every word in the vocabulary
        """
        self.W1 = np.random.rand(self.num_word, self.word_dim) - 0.5 # (num_word, word_dim), uniform distribution from -0.5 - 0.5
        self.W2 = np.random.rand(self.num_word, self.word_dim) - 0.5 # (num_word, word_dim)
        
        # self.SSdW1 = np.zeros(self.W1.shape)# (num_word, word_dim), Sum of Square of W1 derivative matrix
        # self.SSdW2 = np.zeros(self.W2.shape)
        
        self.b1 = np.zeros(self.num_word) # (num_word, 1), bias vector
        self.b2 = np.zeros(self.num_word)
        
        
                            
                            
    def fit(self, 
            corpus, 
            window_size = 5, 
            vocab_limit = 50000,
            x_max = 100,
            alpha = 0.75,
            batch_size = 50,
            learning_rate = 0.1,
            epochs = 10,
            save_vocab=None, 
            load_vocab=None, 
            save_matrix=None, 
            load_matrix=None, 
            save_word_table=None,
            save_word_vectors=None):
        """
        train glove model based on the cooccurrence matrix
        corpus: corpus directory
        window_size: context window size
        vocab_limit: vocabulary limit for cooccurrence matrix due to memory constraint
        x_max: x_max in weighting function of objective function
        alpha: alpha in weighting function of objective function
        batch_size: number of training samples in one mini-batch
        learning_rate: learning rate of gradient update
        epochs: number of iterations for training word vectors
        save_vocab: path of csv file to save vocabulary dictionary
        load_vocab: csv file to load vocabulary dictionary
        save_matrix: path of npy file to save cooccurrence matrix
        load_matrix: npy file to load cooccurrence matrix
        """
        self.corpus = corpus
        self.window_size = window_size
        self.vocab_limit = vocab_limit
        self.x_max = x_max
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_vocab = save_vocab
        self.load_vocab = load_vocab
        self.save_matrix = save_matrix
        self.load_matrix = load_matrix
        self.save_word_table = save_word_table
        self.save_word_vectors = save_word_vectors
        
        # build vocabulary dictionary
        self.vocab_list()
        
        # build cooccurrence matrix
        self.cooccurrence_matrix()
        
        # train word vectors
        self.train()
    

# character level RNN
class RNN(object):
    def __init__(self, hidden_layer_size, activation=Tanh()):
        """
        vanilla RNN implementation
        sizes: a list of size of each layer, including input and output layer
        activations: a list of activation functions corresponding to each layer except for input layer
        cost: cost function (loss, objective)
        """
        self.size = hidden_layer_size
        self.activation = activation
        
    
    def initialize_params(self):
        self.Wh = np.random.randn(self.size, self.size)/100
        self.Wx = np.random.randn(self.size, self.vocab_size)/100
        self.Ws = np.random.randn(self.vocab_size, self.size)/100
        self.bh = np.zeros((self.size, 1)) # hidden bias
        self.bs = np.zeros((self.vocab_size, 1)) # output bias

    
    
    def fit(self, 
            corpus, 
            seq_size = 25,
            learning_rate = 0.1, 
            epochs = 5,
            generate_text_length = 200):
        """
        train vanilla RNN
        corpus: corpus directory
        seq_size: sequence size
        learning_rate: learning rate for gradient descent, 0.01 is a good starting point for Adaptive gradients (Adagrad, Adadelta, Adam, RMSprops, etc.)
        epochs: number of epochs for training
        generate_text_length: text length generated during training, set to 0 to diable it
        """
        self.corpus = corpus
        self.seq_size = seq_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.generate_text_length = generate_text_length
        
        # preprocess files
        text_corpus = self.process(corpus)
                    
        # training
        batch_indices = [i for i in range(0, self.data_size, seq_size)]
        
        losses = []
        for epoch in range(epochs):
            print("Epoch ", epoch)
            counter = 0
            h_prev = np.zeros((self.size, 1)) # reset RNN memory
            
            # sum of squares of gradients for adagrad
            SSdWh, SSdWx, SSdWs = np.zeros_like(self.Wh), np.zeros_like(self.Wx), np.zeros_like(self.Ws)
            SSdbh, SSdbs = np.zeros_like(self.bh), np.zeros_like(self.bs)
            epoch_loss = 0
            
            # mini-batch training
            for i in batch_indices:
                counter += 1
                inputs = [self.char_to_ix[char] for char in text_corpus[i : i+seq_size]] # input character indices
                labels = [self.char_to_ix[char] for char in text_corpus[i+1 : i+seq_size+1]] # target character indices
                
                # generate text after a few batches
                if generate_text_length and counter%500 == 0:
                    text = self.generate_text(h_prev, inputs[0])
                    print("Progress: {0}%", 100*counter/len(batch_indices))
                    print("Last batch loss: ", loss)
                    print("Generating text: ", text)
                
                # forward
                xs, hs, ys, ps = {}, {}, {}, {}
                hs[-1] = np.copy(h_prev) # the initial hidden state is the last hidden state from previous batch
                loss = 0
                for t in range(len(inputs)):
                    xs[t] = np.zeros((self.vocab_size, 1)) # (vocab_size, 1)
                    xs[t][inputs[t]] = 1 # one hot encoding of input character
                    
                    hs[t] = np.tanh(np.dot(self.Wh, hs[t-1]) + np.dot(self.Wx, xs[t]) + self.bh) # hidden state at t (size, 1)
                    ys[t] = np.dot(self.Ws, hs[t]) + self.bs # (vocab_size, 1)
                    ps[t] = softmax(ys[t]) # softmax (vocab_size, 1)
                    loss += -np.log(ps[t][labels[t]])
                
                # update to last hidden state
                h_prev = hs[len(inputs) - 1]
                
                # loss
                epoch_loss += loss
                losses.append(loss)
                
                # backprop
                dWh, dWx, dWs = np.zeros_like(self.Wh), np.zeros_like(self.Wx), np.zeros_like(self.Ws) # init gradients
                dbh, dbs = np.zeros_like(self.bh), np.zeros_like(self.bs)
                dhnext = np.zeros_like(hs[0])
                for t in reversed(range(len(inputs))):
                    dy = np.copy(ps[t]) # (vocab_size, 1)
                    dy[labels[t]] -= 1 # delta 
                    dWs += np.dot(dy, hs[t].T) # output matrix gradient (vocab_size,1) x (1,size) = (vocab_size, size)
                    dbs += dy # output bias gradient (vocab_size, 1)
                    
                    dh = np.dot(self.Ws.T, dy) + dhnext # (size, vocab_size) x (vocab_size, 1) = (size, 1)
                    dhraw = (1 - hs[t] * hs[t]) * dh # tanh_prime(z) = 1 - tanh(z) * tanh(z)
                    
                    dbh += dhraw # (size, 1)
                    dWh += np.dot(dhraw, hs[t-1].T) # (size, 1) x (1, size) = (size, size)
                    dWx += np.dot(dhraw, xs[t].T) # (size, 1) x (1, vocab_size) = (size, vocab_size)
                    
                    dhnext = np.dot(self.Wh, dhraw) # (size, size) x (size, 1) = (size, 1)
                
                # clip gradients to prevent gradient exploding
                for dparam in [dWh, dWx, dWs, dbh, dbs]:
                    np.clip(dparam, -5, 5, out=dparam) 
                
                # gradient update
                for param, dparam, SSdparam in zip([self.Wh, self.Wx, self.Ws, self.bh, self.bs],
                                              [dWh, dWx, dWs, dbh, dbs],
                                              [SSdWh, SSdWx, SSdWs, SSdbh, SSdbs]):
                    SSdparam += dparam * dparam
                    param += -self.learning_rate * dparam/np.sqrt(SSdparam + 1e-8)
            epoch_loss = seq_size*epoch_loss/self.vocab_size
            print("Epoch {0} loss: {1}".format(epoch, epoch_loss))
        print("Loss data per mini batch: ")
        print(losses)
    
    def process(self, corpus):
        self.text_filenames = [join(corpus, f) for f in listdir(corpus) if isfile(join(corpus, f)) and not f.startswith('.')]
        print("List of corpus files: ")
        for filename in self.text_filenames:
            print(filename)
        self.num_files = len(self.text_filenames)
        
        # text data in string format
        data = ""
        for text_filename in self.text_filenames:
            textfile = open(text_filename, "r")
            data += textfile.read()
        
        # a list of unique characters
        self.chars = list(set(data))
        
        self.data_size = len(data)
        self.vocab_size = len(self.chars)
        print("Data size: ", self.data_size)
        print("Vocab size: ", self.vocab_size)
        
        self.char_to_ix = {char:i for i, char in enumerate(self.chars)}
        self.ix_to_char = {i:char for i, char in enumerate(self.chars)}
        
        # init params
        self.initialize_params()
        
        return data

    
    def generate_text(self, h, seed_idx):
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        ixes = []
        
        for t in range(self.generate_text_length):
            h = np.tanh(np.dot(self.Wh, h) + np.dot(self.Wx, x) + self.bh)
            y = np.dot(self.Ws, h) + self.bs
            p = softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return "".join([self.ix_to_char[ix] for ix in ixes])    
        
def softmax(z):
    z = z - np.max(z, axis = 0)
    return np.exp(z)/np.sum(np.exp(z), axis = 0)