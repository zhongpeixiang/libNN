import numpy as np

class L2Regularizer(object):
    def __init__(self, _lambda = 1):
        """
        _lambda: regularization coefficient, larger dataset may need larger _lambda
        """
        self._lambda = _lambda
    
    def evaluate(self, W):
        """
        evaluate regularization cost
        """
        cost = 0
        for layer_W in W:
            cost += 0.5*np.linalg.norm(layer_W)**2
        return self._lambda * cost
    
    
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


class L1Regularizer(object):
    def __init__(self, _lambda = 0.5):
        """
        _lambda: regularization coefficient, larger dataset may need larger _lambda
        """
        self._lambda = _lambda
    
    def evaluate(self, W):
        """
        evaluate regularization cost
        """
        cost = 0
        for layer_W in W:
            cost += np.sum(np.abs(layer_W))
        return self._lambda * cost
    
    def weight_derivative(self, W):
        """
        W: a list of weight matrix comprimsing all weights in the network
        """
        # return derivatives of |W|
        return [2 * (layer_W > 0).astype(np.float) - np.ones(layer_W.shape) for layer_W in W]
    
    def bias_derivative(self, b):
        """
        b: a list of bias vector comprimsing all biases in the network
        """
        return [layer_b * 0 for layer_b in b]