import numpy as np

class SGD(object):
    def __init__(self):
        self.t = 0
    
    def set_sizes(self, sizes):
        pass
        
    def update_W(self, weights, delta_weights, lr):
        """
        weights: weights parameters for one layer (weights or biases)
        delta_weights: delta weights (weight gradients)
        lr: learning rate
        """
        self.t += 1
        return [W - lr*dW for W, dW in zip(weights, delta_weights)]
    
    def update_b(self, biases, delta_biases, lr):
        return [b - lr*db for b, db in zip(biases, delta_biases)]
    


class SGDMomentum(object):
    def __init__(self, gamma=0.9):
        """
        SGD with Momentum
        gamma: momentum term, larger if the gradient has more momentum in original directions and resist new changes
        """
        self.t = 0
        self.gamma = gamma
        
    
    def set_sizes(self, sizes):
        self.sizes = sizes
        # moment
        self.moment_W = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.moment_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
        
    def update_W(self, weights, delta_weights, lr):
        """
        weights: weights parameters for one layer (weights or biases)
        delta_weights: delta weights (weight gradients)
        lr: learning rate
        """
        self.t += 1
        # update moment
        self.moment_W = [self.gamma*mW + lr*dW for mW, dW in zip(self.moment_W, delta_weights)]
        return [W - mW for W, mW in zip(weights, self.moment_W)]
    
    def update_b(self, biases, delta_biases, lr):
        self.moment_b = [self.gamma*mb + lr*db for mb, db in zip(self.moment_b, delta_biases)]
        return [b - mb for b, mb in zip(biases, self.moment_b)]




class Adagrad(object):
    def __init__(self, epsilon=1e-8):
        """
        Adaptive gradient algorithm for each parameter
        epsilon: smoothing term in the denorminator to avoid division by zero
        """
        self.epsilon = epsilon
        self.t = 0
    
    def set_sizes(self, sizes):
        self.sizes = sizes
        # sum of squares of weights and biases
        self.SSdW = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.SSdb = [np.zeros((y, 1)) for y in self.sizes[1:]]
    
    def update_W(self, weights, delta_weights, lr):
        self.t += 1
        self.SSdW = [SSdW + dW**2 for SSdW, dW in zip(self.SSdW, delta_weights)]
        return [W - lr*np.power(SSdW + self.epsilon, -1/2)*dW for W, dW, SSdW in zip(weights, delta_weights, self.SSdW)]
    
    def update_b(self, biases, delta_biases, lr):
        self.SSdb = [SSdb + db**2 for SSdb, db in zip(self.SSdb, delta_biases)]
        return [b - lr*np.power(SSdb + self.epsilon, -1/2)*db for b, db, SSdb in zip(biases, delta_biases, self.SSdb)]
    
    
    
    
class Adadelta(object):
    def __init__(self, epsilon=1e-8, gamma=0.9):
        """
        epsilon: smoothing term in the denorminator to avoid division by zero
        gamma: decay parameter for sum of square of gradients
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.t = 0
    
    def set_sizes(self, sizes):
        """
        SSdW: sum of squares of delta weights
        SSdb: sum of squares of delta biases
        SSdWlr: sum of squares of delta weights with learning rate included
        SSdblr: sum of squares of delta biases with learning rate included
        """
        self.sizes = sizes
        # sum of squares of delta weights and biases
        self.SSdW = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.SSdb = [np.zeros((y, 1)) for y in self.sizes[1:]]
        
        # sum of squares of delta weights and biases, learning rate included
        self.SSdWlr = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.SSdblr = [np.zeros((y, 1)) for y in self.sizes[1:]]
    
    def update_W(self, weights, delta_weights, lr):
        self.t += 1
        # decaying average SSdW
        self.SSdW = [self.gamma*SSdW + (1-self.gamma)*(dW**2) for SSdW, dW in zip(self.SSdW, delta_weights)]
        weights = [W - np.power((SSdWlr + self.epsilon)/(SSdW + self.epsilon), 1/2)*dW 
                   for W, dW, SSdW, SSdWlr in zip(weights, delta_weights, self.SSdW, self.SSdWlr)]
        # decaying average SSdWlr
        self.SSdWlr = [self.gamma*SSdWlr + (1-self.gamma)*(lr*np.power(SSdW + self.epsilon, -1/2)*dW)**2 
                       for SSdWlr, SSdW, dW in zip(self.SSdWlr, self.SSdW, delta_weights)]
        return weights
    
    def update_b(self, biases, delta_biases, lr):
        self.SSdb = [self.gamma*SSdb + (1-self.gamma)*(db**2) for SSdb, db in zip(self.SSdb, delta_biases)]
        biases = [b - np.power((SSdblr + self.epsilon)/(SSdb + self.epsilon), 1/2)*db 
                   for b, db, SSdb, SSdblr in zip(biases, delta_biases, self.SSdb, self.SSdblr)]
        self.SSdblr = [self.gamma*SSdblr + (1-self.gamma)*(lr*np.power(SSdb + self.epsilon, -1/2)*db)**2 
                       for SSdblr, SSdb, db in zip(self.SSdblr, self.SSdb, delta_biases)]
        return biases
    
    
class RMSprop(object):
    def __init__(self, epsilon=1e-8, gamma=0.9):
        """
        epsilon: smoothing term in the denorminator to avoid division by zero
        gamma: decay parameter for sum of square of gradients
        """
        self.epsilon = epsilon
        self.gamma = gamma
        self.t = 0
    
    def set_sizes(self, sizes):
        """
        SSdW: sum of squares of delta weights
        SSdb: sum of squares of delta biases
        SSdWlr: sum of squares of delta weights with learning rate included
        SSdblr: sum of squares of delta biases with learning rate included
        """
        self.sizes = sizes
        # sum of squares of delta weights and biases
        self.SSdW = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.SSdb = [np.zeros((y, 1)) for y in self.sizes[1:]]
    
    def update_W(self, weights, delta_weights, lr):
        self.t += 1
        # decaying average SSdW
        self.SSdW = [self.gamma*SSdW + (1-self.gamma)*(dW**2) for SSdW, dW in zip(self.SSdW, delta_weights)]
        return [W - lr*np.power(SSdW + self.epsilon, -1/2)*dW for W, dW, SSdW in zip(weights, delta_weights, self.SSdW)]
    
    def update_b(self, biases, delta_biases, lr):
        self.SSdb = [self.gamma*SSdb + (1-self.gamma)*(db**2) for SSdb, db in zip(self.SSdb, delta_biases)]
        return [b - lr*np.power(SSdb + self.epsilon, -1/2)*db for b, db, SSdb in zip(biases, delta_biases, self.SSdb)]
    
    
class Adam(object):
    def __init__(self, epsilon=1e-8, beta1=0.9, beta2=0.999):
        """
        epsilon: smoothing term in the denorminator to avoid division by zero
        beta1: first order moment decay parameter
        beta2: second order moment decay parameter
        """
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
    
    def set_sizes(self, sizes):
        """
        SdW: sum of delta weights
        SSdW: sum of squares of delta weights
        Sdb: sum of delta biases
        SSdb: sum of squares of delta biases
        """
        self.sizes = sizes
        # sum of squares of delta weights and biases
        self.SdW = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.SSdW = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.Sdb = [np.zeros((y, 1)) for y in self.sizes[1:]]
        self.SSdb = [np.zeros((y, 1)) for y in self.sizes[1:]]
    
    def update_W(self, weights, delta_weights, lr):
        self.t += 1
        # decaying average SdW, SSdW
        self.SdW = [self.beta1*SdW + (1-self.beta1)*dW for SdW, dW in zip(self.SdW, delta_weights)]
        self.SSdW = [self.beta2*SSdW + (1-self.beta2)*(dW**2) for SSdW, dW in zip(self.SSdW, delta_weights)]
        
        # correct bias
        # print(self.t)
        corrected_SdW = [SdW/(1 - self.beta1**self.t) for SdW in self.SdW]
        corrected_SSdW = [SSdW/(1 - self.beta2**self.t) for SSdW in self.SSdW]
        
        return [W - lr/(np.sqrt(SSdW) + self.epsilon)*dW for W, dW, SSdW in zip(weights, corrected_SdW, corrected_SSdW)]
    
    def update_b(self, biases, delta_biases, lr):
        # decaying average SdW, SSdW
        self.Sdb = [self.beta1*Sdb + (1-self.beta1)*db for Sdb, db in zip(self.Sdb, delta_biases)]
        self.SSdb = [self.beta2*SSdb + (1-self.beta2)*(db**2) for SSdb, db in zip(self.SSdb, delta_biases)]
        
        # correct bias
        corrected_Sdb = [Sdb/(1 - np.power(self.beta1, self.t)) for Sdb in self.Sdb]
        corrected_SSdb = [SSdb/(1 - np.power(self.beta2, self.t)) for SSdb in self.SSdb]
        
        return [b - lr/(np.sqrt(SSdb) + self.epsilon)*db for b, db, SSdb in zip(biases, corrected_Sdb, corrected_SSdb)]
    