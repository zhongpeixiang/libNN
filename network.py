import numpy as np

class Network(object):
    def __init__(self, sizes):
        """
        sizes: a list of size of each layer, including input and output layer
        activation: activation function
        """
        self.length = len(sizes)
        self.sizes = sizes
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        initialize weights and bias for each layer to corresponding size
        """
        self.W = [np.random.randn(y, x)/np.sqrt(784) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
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
        for W, b in zip(self.W, self.b):
            z = np.dot(W, a) + b
            zs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
            
            
        """
        backpropagate errors to previous layers and update weights and biases
        """
        batch_size = self.batch_size
        
        # errors for each layer
        delta = self.cost_derivative(activations[-1], labels) * self.sigmoid_prime(zs[-1])
        
        # weight and bias updates
        delta_weights = [np.zeros(W.shape) for W in self.W]
        delta_biases = [np.zeros(b.shape) for b in self.b]
        
        delta_weights[-1] = np.matmul(delta, activations[-2].T)/batch_size
        delta_biases[-1] = np.mean(delta, axis = 1, keepdims=True)
        

        for i in range(2, self.length):
            delta = np.dot(self.W[-i + 1].T, delta) * self.sigmoid_prime(zs[-i])
            
            # average updates over mini-batch
            delta_weights[-i] = np.dot(delta, activations[-i - 1].T)/batch_size
            delta_biases[-i] = np.mean(delta, axis = 1, keepdims=True)
            
        # update weights and biases
        self.W = [W - self.learning_rate * dW for W, dW in zip(self.W, delta_weights)]
        self.b = [b - self.learning_rate * db for b, db in zip(self.b, delta_biases)]
    
    
    
    
    def fit(self, data, labels, learning_rate = 0.1, epochs = 10, batch_size = 32):
        """
        data: training features, numpy ndarray of size n*F, where n is number of training data and F is number of features in each sample
        labels: training labels, numpy ndarray of size n*1, where n is number of training data
        learning_rate: learning rate for gradient descent
        epochs: number of epochs for training
        batch_size: number of training samples in one batch
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        n = len(data)
        dim_data = data.shape[1]
        dim_labels = labels.shape[1]
        
        # concatenate to one array for shuffling
        data_labels = np.concatenate((data, labels), axis = 1)
        
        for e in range(epochs):
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
        
    def score(self, data, labels):
        predictions = self.predict(data.T)
        true_labels = np.argmax(labels, axis = 1)
        return sum(predictions == true_labels)/len(labels)
    
    
    def predict(self, data):
        # feedforward
        a = data
        for W, b in zip(self.W, self.b):
            a = self.sigmoid(np.dot(W, a) + b)
        print(a)
        return np.argmax(a, axis = 0)
    
    
    def cost_derivative(self, output_activations, labels):
        """
        compute the derivatives of cost with respect to output activations
        """
        return (output_activations - labels)
    
      
    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))
    
    
    
   
        