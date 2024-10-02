import numpy as np


class LinearLayer(object):
    def __init__(self, input_size, output_size, activation, bias=None):
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
    def __init__(self, input_size, output_size, hidden_layers:tuple, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation


    def forward(self, x):
        input_layer = LinearLayer(self.input_size, self.hidden_layers[0], activation=self.activation)
        x = input_layer.forward(x)
        print(input_layer)
        for i in range(len(self.hidden_layers)-1):
            hidden_layer = LinearLayer(self.hidden_layers[i], self.hidden_layers[i+1], activation=self.activation)
            x = hidden_layer.forward(x)
            print(hidden_layer)

        output_layer = LinearLayer(self.hidden_layers[i+1], self.output_size, activation=self.activation)
        x = output_layer.forward(x)
        print(output_layer)
