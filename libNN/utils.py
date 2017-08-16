import sys
import numpy as np
import json

from libNN.network import Network
from libNN.activation import *
from libNN.cost import *

def load(filename):
    print("Loading model from " + filename + " ...")
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    
    # load parameters
    cost = getattr(sys.modules[__name__], data["cost"])()
    activations = [getattr(sys.modules[__name__], a)() for a in data["activations"]]
    model = Network(sizes = data["sizes"], activations = activations, cost = cost)
    model.W = [np.array(W) for W in data["W"]]
    model.b = [np.array(b) for b in data["b"]]
    print("Loading completed")
    
    return model


class BatchNormalization(object):
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
    
    def init_params(self, size):
        self.size = size
        self.gamma = np.ones((size, 1))
        self.beta = np.zeros((size, 1))
        
    def set_lr(self, lr):
        self.lr = lr
    
    def transform(self, z):
        # BN transform
        self.z = z
        self.mean = np.mean(z, axis = 1, keepdims=True)
        self.var = np.var(z, axis = 1, keepdims=True)
        self.z_ = (z - self.mean)/np.sqrt(self.var + self.epsilon)
        self.y = self.gamma * z + self.beta
        
        return self.y
    
    def backprop(self, delta):
        """
        return delta loss with respect to original z
        """
        # batch size
        self.m = delta.shape[1]
        
        delta_z_ = delta * self.gamma # Sl * m
        delta_var = np.sum(-0.5*delta_z_*(self.z - self.mean)*(self.var + self.epsilon)**(-1.5), axis=1, keepdims=True)
        delta_mean = np.sum(-delta_z_/np.sqrt(self.var + self.epsilon), axis = 1, keepdims=True) \
                - 2 * delta_var*np.mean(self.z - self.mean, axis = 1, keepdims=True)
        delta_z = delta_z_/np.sqrt(self.var + self.epsilon) + 2*delta_var*(self.z-self.mean)/self.m + delta_mean/self.m
        delta_gamma = np.sum(delta * self.z_, axis=1, keepdims=True)
        delta_beta = np.sum(delta, axis=1, keepdims=True)
        
        # update gamma and beta
        self.gamma -= self.lr * delta_gamma
        self.beta -= self.lr * delta_beta
        
        return delta_z
    
    def prediction_transform(self, z):
        """
        transform during predictions
        """
        pre_mean = np.mean(z, axis = 1, keepdims = True)
        pre_var = self.m * np.var(z, axis = 1, keepdims = True)/(self.m - 1)
        pre_y = self.gamma*z/np.sqrt(pre_var + self.epsilon) + (self.beta - self.gamma*pre_mean/np.sqrt(pre_var + self.epsilon))
        return pre_y