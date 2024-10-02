import numpy as np


class LinearLayer(object):
    def __init__(self, x, input_size, output_size, activation_function, bias=None):
        self.x = x
        self.weights = np.random.randn(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.bias = bias

    def forward(self):
        output = np.dot(self.x, self.weights)
        if not self.bias is None: output = output + self.bias
        output = self.activation_function(output)
        return(output)

