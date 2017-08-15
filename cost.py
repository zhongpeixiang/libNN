import numpy as np

class QuadraticCost(object):
    def __init__(self):
        pass
    
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
    
    def delta(self, z, y, g):
        a = g.evaluate(z)
        if str(g.__class__.__name__) == "Sigmoid":
            return (a - y)
        return (a - y)/(a * (1-a)) * g.prime(z)
    
    