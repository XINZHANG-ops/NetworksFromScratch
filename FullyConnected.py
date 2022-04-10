import numpy as np

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

