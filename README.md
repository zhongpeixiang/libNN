# libNN
#### A (deep) neural network library built from scratch in Python 3



## Dependencies
Python3, Numpy, Matplotlib (optional)



## Installation
Clone this git repo and run run.py to get started, the data is MNIST handwritten digit recognition data set



## Currently Implemented: 

### Network
1. Fully-connected feedforward neural network
2. Word2Vec (skipgram and cbow)
3. GloVe


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
4. Early stopping


### Others
1. Batch normalization



## References
1. Michael A. Nielsen, "Neural Networks and Deep Learning", Determination Press, 2015.(http://neuralnetworksanddeeplearning.com/)
2. Ruder, Sebastian. "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747 (2016). (https://arxiv.org/pdf/1609.04747.pdf)
3. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International Conference on Machine Learning. 2015. (http://proceedings.mlr.press/v37/ioffe15.pdf)
4. Srivastava, Nitish, et al. "Dropout: a simple way to prevent neural networks from overfitting." Journal of machine learning research 15.1 (2014): 1929-1958. (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
5. Prechelt, Lutz. "Early stopping-but when?." Neural Networks: Tricks of the trade (1998): 553-553. (https://www.researchgate.net/profile/Lutz_Prechelt/publication/2874749_Early_Stopping_-_But_When/links/551bc1650cf2fe6cbf75e533.pdf)
6. Stanford Online Course CS224n: Natural Language Processing with Deep Learning (http://web.stanford.edu/class/cs224n/index.html)
7. Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781 (2013). (https://arxiv.org/pdf/1301.3781.pdf)
8. Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013. (http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
9. Stanford Online Course CS224n Solutions (https://github.com/kvfrans/cs224-solutions)
10. Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.
11. GloVe Github Repository (https://github.com/stanfordnlp/GloVe)