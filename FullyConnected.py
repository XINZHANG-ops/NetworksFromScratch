import numpy as np

class LabelTypeError(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message

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
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on input values
        self.dinputs = np.dot(dvalues, self.weights.T)



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
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

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

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)


# Loss
# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class CrossEntropyLoss(Loss):
    """
    we can look up formula for this, if it is a binary class,
    the formula is: −(y * log(p)+(1−y) * log(1−p)),
    and for multiple class, the formula is similar, an example will make it clear.

    Example:
    assume we have 3 classes, and for one data point, the true one hot label is [1, 0, 0],
    which means the first class is the ground truth.

    and our output from softmax is [0.7, 0.2, 0.1], then our CrossEntropyLoss is simply as follow:

    - (1 * log(0.7) + 0 * log(0.1) + 0 * log(0.2)) = 0.35667494393873245, where log is natural log,
    note that log(x) is a negative value when x is in (0, 1), and log(x) -> -∞ as x -> 0

    Note:
    due to our one hot label is always like one index in 1, all other indices are 0, thus we
    can simplify our calculation as -log(p) where p is probability in softmax prediction with the same
    index as 1 in true label.

    Caution:
    if we are dealing with multi-label classification, the above simplification is no-longer valid, then
    we need to calculate CrossEntropyLoss follow the formula
    """
    def forward(self, input, labels):
        """
        This will calculate the averaged CrossEntropyLoss for a batch

        :param input: a batch of model raw predictions, expect probabilities
        :param labels: one hot labels or categorical labels
        example:
        obe hot labels:
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0]])

        same labels in categorical format:
        np.array([0, 1, 1])

        :return:
        """
        epsilon = 1e-7 # avoid log(0) in loss calculation
        labels = np.array(labels)
        if len(labels.shape) == 1:
            # this simply select the confidence from labels using labels as indices
            """
            for example:
            input = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
            labels = [0, 1, 1]
            
            input[range(len(labels)), labels] ==> array([0.7, 0.5, 0.9])
            
            """
            target_index_conf = input[range(len(input)), labels]

        elif len(labels.shape) == 2: # this case we are receiving one hot encoding labels
            """
            for example:
            input = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
            labels = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])
                          
            np.sum(input * labels, axis=1) ==> array([0.7, 0.5, 0.9])

            """
            target_index_conf = np.sum(input * labels, axis=1)
        else:
            raise LabelTypeError('Labels must be Categorical or One hot')

        # np.clip will make sure target_index_conf are in range 1e-7 and 1-1e-7
        # for values < 1e-7, will be clipped to 1e-7, for value > 1-1e-7 will be clipped to 1-1e-7
        target_index_conf = np.clip(target_index_conf, 1e-7, 1 - 1e-7)
        return -np.log(target_index_conf)

    def backward(self, input, labels):
        samples = len(input) # bacth size
        cls_num = len(input[0]) # number of classes

        # If labels are categorical, turn them into one-hot vector
        if len(labels.shape) == 1:
            labels = np.eye(cls_num)[labels]

        # find gradients
        self.dinputs = - labels / input
        # normalize gradient
        self.dinputs /= samples
