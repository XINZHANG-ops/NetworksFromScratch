import numpy as np


# Layers

class Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        :param n_inputs: int, input size
        :param n_neurons: int, layer size
        """

        """
        input will be a matrix, where each row with size n_inputs, each column with batch_size
        input will have shape batch_size * n_inputs
        input will multiply weights matrix, weights matrix has shape n_inputs * n_neurons
        
        np.random.randn generate normal distribution with mean 0, std 1
        normally, start weights with small enourght but not 0 value will have better convergence time
        so we multiply by 0.01
        """

        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # we initialize biases as zeros
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Activation function
"""
The reason we need non-linear activation function is:
If we don't have non-linear activations, our output will always be a linear function of input

Let say we have 3 layers, with weight matrix W1, W2, W3, where W1 with shape n*h1, W2 with shape h1*h2, W3 with shape h2*h3

the output = X * W1 * W2 * W3, lets say W1 * W2 * W3 = W, thus output = X * W, which means deeper layers 
make no difference as we put just one layer, since the linearity.

"""

class ReLu:
    def forward(self, inputs):
        """

        :param inputs: output from previous layer, should be b * h matrix, where b is batch_size,
                        h is the size of previous layer
        :return:
        """
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        """

        :param inputs: inputs: output from previous layer, should be b * h matrix, where b is batch_size,
                        h is the size of previous layer
        :return:
        """
        # raw probabilities
        """
        keepdims=True will keep original dimension, for example if inputs with shape (b, h), 
        np.max(inputs, axis=1, keepdims=True) with shape (b, 1), otherwise if
        np.max(inputs, axis=1, keepdims=False) with shape (b, ), which cannot be subtracted from inputs
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # nomarlized probabilities
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities