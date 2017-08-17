# libNN
#### A (deep) neural network library built from scratch in Python 3


<br>
## Dependencies
Python3, Numpy, Matplotlib (optional)


<br>
## Installation
Clone this git repo and run run.py to get started, the data is MNIST handwritten digit recognition data set


<br>
## Currently Implemented: 

### Network
1. Fully-connected feedforward neural network


### Weight Initialization
1. Normalized by input dimension to avoid gradient satuaration at the start of training


### Activation Functions
1. Sigmoid
2. Tanh
3. ReLU


### Cost Functions
1. Quadratic cost
2. Cross-entropy


### Optimizer
1. SGD
2. SGDMomentum
3. Adagrad
4. Adadelta
5. RMSprop
6. Adam


### Regularizer
1. L1 regularizer
2. L2 regularizer
3. Dropout


### Others
1. Batch normalization


<br>
## References
1. Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015.(http://neuralnetworksanddeeplearning.com/)
2. Ruder, Sebastian. "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747 (2016). (https://arxiv.org/pdf/1609.04747.pdf)
3. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International Conference on Machine Learning. 2015. (http://proceedings.mlr.press/v37/ioffe15.pdf)
4. Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." Journal of machine learning research 15.1 (2014): 1929-1958. (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)