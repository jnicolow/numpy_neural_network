import numpy as np


class LinearLayer(object):
    def __init__(self, input_size, output_size, activation, weight_init=None, bias=None):
        if not weight_init is None:
            self.weights = weight_init((input_size, output_size))
        else:
            self.weights = np.random.randn(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.bias = bias

    def __repr__(self):
        return f"LinearLayer input:{self.input_size}, output:{self.output_size}, activation:{str(self.activation)}"


    def forward(self, x):
        output = np.dot(x, self.weights)
        if not self.bias is None: output = output + self.bias
        output = self.activation(output)
        return(output)


class SequentialModel(object):
    def __init__(self, input_size:int, output_size:int, hidden_layers:tuple, activations:tuple, weight_init:tuple=None):
        self.input_size = input_size
        # self.output_size = output_size
        self.layers = (input_size, *hidden_layers, output_size) # all the layers but the iput layer
        print(self.layers)
        self.activations = activations
        if weight_init is None: self.weight_init = (None,) * len(activations)  # create tuple of None to just use random
        else: self.weight_init = weight_init  # use the provided weight_init if it exists



    def forward(self, x):

        for i in range(1, len(self.layers)):

            layer = LinearLayer(
                self.layers[i-1], 
                self.layers[i], 
                activation=self.activations[i], 
                weight_init=self.weight_init[i]
            )
            x = layer.forward(x)
            print(layer)

        return x
    
    def backward(self, x, y):
        # performs a backward pass through the model
        


