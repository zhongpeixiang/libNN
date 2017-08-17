import numpy as np

class QuadraticCost(object):
    def __init__(self):
        pass
    
    
    def evaluate(self, a, y):
        """
        a: output activation or model predictions
        y: label
        return total cost between a and y
        """
        return 0.5*np.linalg.norm(a - y)**2
    
    def delta(self, z, y, g):
        """
        z: weighted input
        y: label
        g: activation function, implementing evaluate(z) and prime(z)
        """
        a = g.evaluate(z)
        return (a - y) * g.prime(z)

class CrossEntropyCost(object):
    def __init__(self):
        pass
    
    
    def evaluate(self, a, y):
        """
        a: output activation or model predictions
        y: label
        return total cost between a and y
        """
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
    
    
    def delta(self, z, y, g):
        a = g.evaluate(z)
        if str(g.__class__.__name__) == "Sigmoid":
            return (a - y)
        return (a - y)/(a * (1-a)) * g.prime(z)
    
    