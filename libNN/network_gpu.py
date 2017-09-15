from os import listdir
from os.path import isfile, join
from time import time

import numpy as np
import torch

class RNN(object):
    def __init__(self, hidden_layer_size):
        """
        vanilla RNN implementation
        hidden_layer_size: number of neurons in hidden layer
        """
        self.size = hidden_layer_size
        
    
    def initialize_params(self):
        # GPU 
        self.Wh = (torch.randn(self.size, self.size)/100).cuda()
        self.Wx = (torch.randn(self.size, self.vocab_size)/100).cuda()
        self.Ws = (torch.randn(self.vocab_size, self.size)/100).cuda()
        self.bh = torch.zeros((self.size, 1)).cuda() # hidden bias
        self.bs = torch.zeros((self.vocab_size, 1)).cuda() # output bias

    
    
    def fit(self, 
            corpus, 
            seq_size = 25,
            learning_rate = 0.01, 
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
            h_prev = torch.zeros((self.size, 1)).cuda() # reset RNN memory
            
            # sum of squares of gradients for adagrad
            SSdWh, SSdWx, SSdWs = torch.zeros(self.Wh.size()).cuda(), torch.zeros(self.Wx.size()).cuda(), torch.zeros(self.Ws.size()).cuda()
            SSdbh, SSdbs = torch.zeros(self.bh.size()).cuda(), torch.zeros(self.bs.size()).cuda()
            epoch_loss = 0
            
            # mini-batch training
            for i in batch_indices:
                counter += 1
                inputs = [self.char_to_ix[char] for char in text_corpus[i : i+seq_size]] # input character indices
                labels = [self.char_to_ix[char] for char in text_corpus[i+1 : i+seq_size+1]] # target character indices
                
                # generate text after a few batches
                if generate_text_length and counter%5000 == 0:
                    text = self.generate_text(h_prev, inputs[0])
                    print("-------------------------------------------------")
                    print("Counter: ", counter)
                    print("Progress: {0}%".format(100*counter/len(batch_indices)))
                    print("Last batch loss: ", loss)
                    print("Generating text: ", text)
                
                # forward
                xs, hs, ys, ps = {}, {}, {}, {}
                hs[-1] = h_prev.clone() # the initial hidden state is the last hidden state from previous batch
                loss = 0
                for t in range(len(inputs)):
                    xs[t] = torch.zeros((self.vocab_size, 1)).cuda() # (vocab_size, 1)
                    xs[t][inputs[t]] = 1 # one hot encoding of input character
                    
                    hs[t] = torch.tanh(torch.mm(self.Wh, hs[t-1]) + torch.mm(self.Wx, xs[t]) + self.bh) # hidden state at t (size, 1)
                    ys[t] = torch.mm(self.Ws, hs[t]) + self.bs # (vocab_size, 1)
                    ps[t] = softmax(ys[t]) # softmax (vocab_size, 1)
                    loss += -torch.log(ps[t][labels[t]])
                
                # update to last hidden state
                h_prev = hs[len(inputs) - 1]
                
                # loss
                epoch_loss += loss
                losses.append(loss)
                
                # backprop
                dWh, dWx, dWs = torch.zeros(self.Wh.size()).cuda(), torch.zeros(self.Wx.size()).cuda(), torch.zeros(self.Ws.size()).cuda() # init gradients
                dbh, dbs = torch.zeros(self.bh.size()).cuda(), torch.zeros(self.bs.size()).cuda()
                dhnext = torch.zeros(hs[0].size()).cuda()
                for t in reversed(range(len(inputs))):
                    dy = ps[t].clone() # (vocab_size, 1)
                    dy[labels[t]] -= 1 # delta error
                    dWs += torch.mm(dy, hs[t].t()) # output matrix gradient (vocab_size,1) x (1,size) = (vocab_size, size)
                    dbs += dy # output bias gradient (vocab_size, 1)
                    
                    dh = torch.mm(self.Ws.t(), dy) + dhnext # (size, vocab_size) x (vocab_size, 1) = (size, 1)
                    dhraw = (1 - hs[t] * hs[t]) * dh # tanh_prime(z) = 1 - tanh(z) * tanh(z)
                    
                    dbh += dhraw # (size, 1)
                    dWh += torch.mm(dhraw, hs[t-1].t()) # (size, 1) x (1, size) = (size, size)
                    dWx += torch.mm(dhraw, xs[t].t()) # (size, 1) x (1, vocab_size) = (size, vocab_size)
                    
                    dhnext = torch.mm(self.Wh, dhraw) # (size, size) x (size, 1) = (size, 1)
                
                # clip gradients to prevent gradient exploding
                for dparam in [dWh, dWx, dWs, dbh, dbs]:
                    dparam = torch.Tensor(np.clip(dparam.cpu().numpy(), -5, 5)).cuda()
                    
                
                # gradient update
                for param, dparam, SSdparam in zip([self.Wh, self.Wx, self.Ws, self.bh, self.bs],
                                              [dWh, dWx, dWs, dbh, dbs],
                                              [SSdWh, SSdWx, SSdWs, SSdbh, SSdbs]):
                    SSdparam += dparam * dparam
                    param += -self.learning_rate * dparam/torch.sqrt(SSdparam + 1e-8)
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
        x = torch.zeros((self.vocab_size, 1)).cuda()
        x[seed_idx] = 1
        ixes = []
        
        for t in range(self.generate_text_length):
            h = torch.tanh(torch.mm(self.Wh, h) + torch.mm(self.Wx, x) + self.bh)
            y = torch.mm(self.Ws, h) + self.bs
            p = softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.cpu().numpy().ravel())
            x = torch.zeros((self.vocab_size, 1)).cuda()
            x[ix] = 1
            ixes.append(ix)
        return "".join([self.ix_to_char[ix] for ix in ixes])    
      
        
        
"""
Deep RNN with multuple layers
"""
class DeepRNN(object):
    def __init__(self, hidden_layer_sizes):
        """
        vanilla RNN implementation
        hidden_layer_sizes: list, number of neurons in each hidden layer
        """
        self.sizes = hidden_layer_sizes
        
    
    def initialize_params(self):
        # GPU with multiple layers
        self.Wh = [(torch.randn(size, size)/100).cuda() for size in self.sizes]
        self.Wx = [(torch.randn(size, self.vocab_size)/100).cuda() for size in self.sizes]
        self.Ws = [(torch.randn(self.vocab_size, size)/100).cuda() for size in self.sizes]
        self.bh = [torch.zeros((size, 1)).cuda() for size in self.sizes]
        self.bs = [torch.zeros((self.vocab_size, 1)).cuda() for size in self.sizes]

    
    
    def fit(self, 
            corpus, 
            seq_size = 25,
            learning_rate = 0.01, 
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
            h_prev = [torch.zeros((size, 1)).cuda() for size in self.sizes] # reset RNN memory
            
            # sum of squares of gradients for adagrad
            SSdWh = [torch.zeros(Wh.size()).cuda() for Wh in self.Wh]
            SSdWx = [torch.zeros(Wx.size()).cuda() for Wx in self.Wx]
            SSdWs = [torch.zeros(Ws.size()).cuda() for Ws in self.Ws]
            SSdbh = [torch.zeros(bh.size()).cuda() for bh in self.bh]
            SSdbs = [torch.zeros(bs.size()).cuda() for bs in self.bs]
            epoch_loss = 0
            
            # mini-batch training
            for i in batch_indices:
                counter += 1
                inputs = [self.char_to_ix[char] for char in text_corpus[i : i+seq_size]] # input character indices
                labels = [self.char_to_ix[char] for char in text_corpus[i+1 : i+seq_size+1]] # target character indices
                
                # generate text after a few batches
                if generate_text_length and counter%5000 == 0:
                    text = self.generate_text(h_prev, inputs[0])
                    print("-------------------------------------------------")
                    print("Counter: ", counter)
                    print("Progress: {0}%".format(100*counter/len(batch_indices)))
                    print("Loss: ", epoch_loss)
                    print("Generating text: ", text)
                
                # forward
                xs, hs, ys = [{} for size in self.sizes], [{} for size in self.sizes], [{} for size in self.sizes]
                ps = {}
                for ix, layer_h_prev in enumerate(h_prev):
                    hs[ix][-1] = layer_h_prev.clone()
                # hs[0][-1] = h_prev.clone() # the initial hidden state is the last hidden state from previous batch
                loss = 0
                for t in range(len(inputs)):
                    xs[0][t] = torch.zeros((self.vocab_size, 1)).cuda() # (vocab_size, 1)
                    xs[0][t][inputs[t]] = 1 # one hot encoding of input character
                    
                    # iterate through each layer
                    """
                    for layer_ix, (hs, ys, Wh, Wx, Ws, bh, bs) in enumerate(zip(hs, ys, self.Wh, self.Wx, self.Ws, self.bh, self.bs)):
                        hs[t] = torch.tanh(torch.mm(Wh, hs[t-1]) + torch.mm(Wx, xs[layer_ix][t]) + bh) # hidden state at t (size, 1)
                        ys[t] = torch.mm(Ws, hs[t]) + bs # (vocab_size, 1)
                        if layer_ix < len(self.sizes) - 1:
                            xs[layer_ix + 1][t] = ys[t] # the output y from current layer is input x for next layer
                    """
                    for k in range(len(self.sizes)):
                        hs[k][t] = torch.tanh(torch.mm(self.Wh[k], hs[k][t-1]) + torch.mm(self.Wx[k], xs[k][t]) + self.bh[k]) # hidden state at t (size, 1)
                        ys[k][t] = torch.mm(self.Ws[k], hs[k][t]) + self.bs[k] # (vocab_size, 1)
                        if k < len(self.sizes) - 1:
                            xs[k + 1][t] = ys[k][t] # the output y from current layer is input x for next layer
                    
                    
                    ps[t] = softmax(ys[len(self.sizes) - 1][t]) # softmax (vocab_size, 1)
                    loss += -torch.log(ps[t][labels[t]])
                
                # update to last hidden state
                h_prev = [layer_hs[len(inputs) - 1] for layer_hs in hs]
                
                # loss
                epoch_loss = 0.999*epoch_loss + 0.001 * loss
                losses.append(loss)
                
                # backprop through multiple layers
                dWh = [torch.zeros(Wh.size()).cuda() for Wh in self.Wh]
                dWx = [torch.zeros(Wx.size()).cuda() for Wx in self.Wx]
                dWs = [torch.zeros(Ws.size()).cuda() for Ws in self.Ws]
                dbh = [torch.zeros(bh.size()).cuda() for bh in self.bh]
                dbs = [torch.zeros(bs.size()).cuda() for bs in self.bs]
                dhnext = [torch.zeros(layer_hs[0].size()).cuda() for layer_hs in hs]
                
                for t in reversed(range(len(inputs))):
                    dy = ps[t].clone() # (vocab_size, 1)
                    dy[labels[t]] -= 1 # delta error at output layer
                    
                    # backprop through each layer
                    """
                    for hs, ys, Wh, Wx, Ws, bh, bs, dWh, dWx, dWs, dbh, dbs, dhnext in reversed(list(zip(hs, ys, self.Wh, self.Wx, self.Ws, self.bh, self.bs, dWh, dWx, dWs, dbh, dbs, dhnext))):
                        
                        dWs += torch.mm(dy, hs[t].t()) # output matrix gradient (vocab_size,1) x (1,size) = (vocab_size, size)
                        dbs += dy # output bias gradient (vocab_size, 1)

                        dh = torch.mm(Ws.t(), dy) + dhnext # (size, vocab_size) x (vocab_size, 1) = (size, 1)
                        dhraw = (1 - hs[t] * hs[t]) * dh # tanh_prime(z) = 1 - tanh(z) * tanh(z)

                        dbh += dhraw # (size, 1)
                        dWh += torch.mm(dhraw, hs[t-1].t()) # (size, 1) x (1, size) = (size, size)
                        dWx += torch.mm(dhraw, xs[t].t()) # (size, 1) x (1, vocab_size) = (size, vocab_size)

                        dhnext = torch.mm(Wh, dhraw) # (size, size) x (size, 1) = (size, 1)
                        
                        # the delta x is delta y at previous layer
                        dy = torch.mm(Ws, dhraw) # (vocab_size, size) x (size, 1) = (vocab_size, 1)
                    """
                    for k in reversed(range(len(self.sizes))):
                        dWs[k] += torch.mm(dy, hs[k][t].t()) # output matrix gradient (vocab_size,1) x (1,size) = (vocab_size, size)
                        dbs[k] += dy # output bias gradient (vocab_size, 1)

                        dh = torch.mm(self.Ws[k].t(), dy) + dhnext[k] # (size, vocab_size) x (vocab_size, 1) = (size, 1)
                        dhraw = (1 - hs[k][t] * hs[k][t]) * dh # tanh_prime(z) = 1 - tanh(z) * tanh(z)

                        dbh[k] += dhraw # (size, 1)
                        dWh[k] += torch.mm(dhraw, hs[k][t-1].t()) # (size, 1) x (1, size) = (size, size)
                        dWx[k] += torch.mm(dhraw, xs[k][t].t()) # (size, 1) x (1, vocab_size) = (size, vocab_size)

                        dhnext[k] = torch.mm(self.Wh[k], dhraw) # (size, size) x (size, 1) = (size, 1)
                        
                        # the delta x is delta y at previous layer
                        dy = torch.mm(self.Ws[k], dhraw) # (vocab_size, size) x (size, 1) = (vocab_size, 1)
                    
                # clip gradients to prevent gradient exploding
                for dparam in [dWh, dWx, dWs, dbh, dbs]:
                    # loop through each layer
                    for k in range(len(self.sizes)):
                        dparam[k] = torch.Tensor(np.clip(dparam[k].cpu().numpy(), -5, 5)).cuda()
                    
                
                # gradient update
                for param, dparam, SSdparam in zip([self.Wh, self.Wx, self.Ws, self.bh, self.bs],
                                              [dWh, dWx, dWs, dbh, dbs],
                                              [SSdWh, SSdWx, SSdWs, SSdbh, SSdbs]):
                    for i in range(len(self.sizes)):
                        SSdparam[i] += dparam[i] * dparam[i]
                        param[i] += -self.learning_rate * dparam[i]/torch.sqrt(SSdparam[i] + 1e-8)
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
        x = torch.zeros((self.vocab_size, 1)).cuda()
        x[seed_idx] = 1
        ixes = []
        
        for t in range(self.generate_text_length):
            for k in range(len(self.sizes)):
                h[k] = torch.tanh(torch.mm(self.Wh[k], h[k]) + torch.mm(self.Wx[k], x) + self.bh[k])
                y = torch.mm(self.Ws[k], h[k]) + self.bs[k]
                x = y
            p = softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.cpu().numpy().ravel())
            x = torch.zeros((self.vocab_size, 1)).cuda()
            x[ix] = 1
            ixes.append(ix)
        return "".join([self.ix_to_char[ix] for ix in ixes])    
    
    
def softmax(z):
    z = z - torch.max(z, dim = 0)[0]
    return torch.exp(z)/torch.sum(torch.exp(z), dim = 0)