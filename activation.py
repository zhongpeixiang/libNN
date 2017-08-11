import numpy as np

class Sigmoid(object):
    def __init__(self):
        pass
    
    def evaluate(self, z):
        return 1.0/(1.0 + np.exp(-z))
    
    def prime(self, z):
        return self.evaluate(z) * (1 - self.evaluate(z))

class ReLU(object):
    def __init__(self):
        pass
    
    def evaluate(self, z):
        return np.maximum(0, z)
    
    def prime(self, z):
        return (z > 0).astype(float)

class Tanh(object):
    def __init__(self):
        pass
    
    def evaluate(self, z):
        return np.tanh(z)
    
    def prime(self, z):
        return 1 - np.tanh(z) ** 2
    
class Softmax(object):
    def __init__(self):
        pass
    
    def evaluate(self, z):
        z = z - np.max(z, axis = 0)
        return np.exp(z)/np.sum(np.exp(z), axis = 0)
         
    def prime(self, z):
        # not fully implemented yet
        a = self.evaluate(z)
        return np.diag(np.sum(a, axis = 1)) - np.dot(a, a.T)