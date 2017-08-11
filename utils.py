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