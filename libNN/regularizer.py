import numpy as np

class L2Regularizer(object):
    def __init__(self):
        pass
    
    def weight_derivative(self, W):
        """
        W: a list of weight matrix comprimsing all weights in the network
        """
        return W
    
    def bias_derivative(self, b):
        """
        b: a list of bias vector comprimsing all biases in the network
        """
        return [layer_b * 0 for layer_b in b]