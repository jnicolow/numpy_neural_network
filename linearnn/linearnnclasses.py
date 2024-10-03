import numpy as np


class LinearLayer(object):
    def __init__(self, input_size, output_size, activation_fn, weight_init=None):
        if not weight_init is None:
            self.weights = weight_init((input_size, output_size))
        else:
            self.weights = np.random.randn(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fn = activation_fn

        self.bias = np.zeros(output_size) # initialized to zero (changed during training)

    def __repr__(self):
        return f"LinearLayer input:{self.input_size}, output:{self.output_size}, activation:{str(self.activation_fn)}"


    def forward(self, x, backprop=False):
        
        linear_transformation_output = np.dot(x, self.weights) + self.bias # Wx + b
        if backprop:self.linear_transrom = linear_transformation_output # saving these to be used in backpropigation

        activation = self.activation_fn(linear_transformation_output) # activation function
        if backprop:self.activation = activation
        return(activation)


class SequentialModel(object):
    def __init__(self, input_size:int, output_size:int, hidden_layers:tuple, activation_fns:tuple, weight_init:tuple=None, loss_fn=None):
        self.input_size = input_size
        # self.output_size = output_size
        self.layers = (input_size, *hidden_layers, output_size) # all the layers but the iput layer
        print(self.layers)
        self.activation_fns = activation_fns
        if weight_init is None: self.weight_init = (None,) * len(activation_fns)  # create tuple of None to just use random
        else: self.weight_init = weight_init  # use the provided weight_init if it exists
        self.loss_fn = loss_fn

        self.model = self.build_model()


    def build_model(self):
        print('building model...')
        model_layers = []
        for i in range(1, len(self.layers)):

            layer = LinearLayer(
                self.layers[i-1], 
                self.layers[i], 
                activation_fn=self.activation_fns[i], 
                weight_init=self.weight_init[i]
            )
            
            print(f'Adding: {layer}')
            model_layers.append(layer)
        print('model built')
        return model_layers



    def forward(self, x, backprop=False):

        for i in range(0, len(self.model)):

            layer = self.model[i]
            x = layer.forward(x, backprop) # if back prop true linear layer will remeber linear transform and activation output

        return x
    

    def backward(self, x, y):
        # performs a backward pass through the model
        y_hat = self.forward(x, backprop=True) # backprop true to save self.linear_transform and self.activation in each layer

        loss = self.loss_fn(y=y, y_hat=y_hat)
        print(loss)
        loss_gradient = None # not sure how to calculate this yet
        

        # layer_gradients = []
        # # start from the last layer of the model
        for i in reversed(range(len(self.model))):
            layer = self.model[i]
            print(layer)

            # d/dx w.r.t. activation
            if i == len(self.model) - 1:
                # this is the last layer so do element wise multiplication (not dot product)
                activation_gradient = self.activation_fns[i].derivative(y_hat) 
                layer_output_gradient = loss_gradient * activation_gradient
            else:
                # for hidden layers
                activation_gradient = self.activation_fns[i].derivative('output from prev layer') # output from previous layer
                layer_output_gradient = np.dot(self.model[i-1].weights.T, layer_output_gradient) * activation_gradient
            


            # d/dx w.r.t. weights and bias

            



