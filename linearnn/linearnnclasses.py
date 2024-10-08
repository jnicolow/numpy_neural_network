import numpy as np
from linearnn.activationfunctions import Softmax, DummyActivation # check if it is softmax because we dont differentiate this activation
# from linearnn.optim import AdamOptimizer # import optimizer incase one isnt provided





class LinearLayer(object):
    def __init__(self, input_size, output_size, activation_fn_class, weight_init=None, learning_rate=0.001, optimizer=None):
        if not weight_init is None:
            self.weights = weight_init((input_size, output_size))
        else:
            self.weights = np.random.randn(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fn_class = activation_fn_class

        self.bias = np.zeros(output_size) # initialized to zero (changed during training)
        self.learning_rate = learning_rate
        if optimizer is None: self.optimizer_weights = None # used in back prop
        else:
            self.optimizer_weights = optimizer() # this is initializing a new optimizer (one specifically for this layer)
            self.optimizer_bias = optimizer() # need a seperate one for weights and bias because they have different sizes

    def __repr__(self):
        return f"LinearLayer input:{self.input_size}, output:{self.output_size}, activation:{str(self.activation_fn_class)}"


    def forward(self, x, backprop=False):
        # if backprop=False info used in back propigation wont be updated allowing for inference during training

        if backprop: self.input = x # remeber input to the layer for backprop

        # linear transformation
        linear_transformation_output = np.dot(x, self.weights) + self.bias # Wx + b
        if backprop:self.linear_transrom = linear_transformation_output # saving these to be used in backpropigation

        # activation
        activation = self.activation_fn_class().forward(linear_transformation_output) # activation function

        if backprop: self.activation = activation

        return(activation)
    

    def backward(self, output_gradient):
        # forward is run inside the sequential model class below (with backprop=True) y only used for differentiating soft max
        # print(self.linear_transrom.shape)

        # d/dx activation function w.r.t. linear transformation Wx + b
        activation_gradient = self.activation_fn_class().derivative(self.linear_transrom) # dL/d activation
        layer_output_gradient = output_gradient * activation_gradient # dL/dz = dL/da * da/dz

        # weights and bias
        # print(self.input.shape)
        # print(layer_output_gradient.shape)
        weights_gradient = np.dot(self.input.T, layer_output_gradient) # dL/dW = X^T * dL/dz
        bias_gradient = np.sum(layer_output_gradient, axis=0)  # dL/db = sum(dL/z)
        
        # make sure gradients dont expload (with out this gradient explosion occured)
        max_norm = 60.0  # normalize to threshold
        total_norm = np.linalg.norm(weights_gradient)
        if total_norm > max_norm:
            scale = max_norm / total_norm
            weights_gradient *= scale
            bias_gradient *= scale

        # print("Max gradient:", np.max(weights_gradient))
        # print("Min gradient:", np.min(weights_gradient))
        # make update to weights and biases
        if self.optimizer_weights is None:
            # just do vanila GD update
            self.weights -= self.learning_rate * weights_gradient / self.input.shape[0] # devide by batch size to get average for batch size
            self.bias -= self.learning_rate * bias_gradient / self.input.shape[0] 
        else:
            self.weights = self.optimizer_weights.update(self.weights, weights_gradient)
            self.bias = self.optimizer_bias.update(self.bias, bias_gradient)
        
        # print(np.dot(layer_output_gradient, self.weights.T).shape)
        return np.dot(layer_output_gradient, self.weights.T) # return the gradient (to be passed to previous layer) (batch size by # nodes in layer)


class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask=None

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) # create mask that drops out certain amount of neurons
            return x * self.mask / (1 - self.dropout_rate) # scale activation 
        else: return x # no drop out during inference

    def backward(self, output_gradient):
        # gradient only goes through non "droped out" nodes
        return output_gradient * self.mask / (1 - self.dropout_rate)



class SequentialModel(object):
    def __init__(self, input_size:int, output_size:int, hidden_layers:tuple, activation_fn_classes:tuple, weight_init:tuple=None, loss_fn_class=None, learning_rate=0.01, optimizer=None):
        self.input_size = input_size
        # self.output_size = output_size
        self.layers = (input_size, *hidden_layers, output_size) # all the layers but the iput layer
        self.activation_fn_classes = activation_fn_classes
        if weight_init is None: self.weight_init = (None,) * len(activation_fn_classes)  # create tuple of None to just use random
        else: self.weight_init = weight_init  # use the provided weight_init if it exists
        self.loss_fn_class = loss_fn_class
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.model = self.build_model() # build the model


    def build_model(self):
        print('building model...')
        model_layers = []
        print(len(self.layers))
     
        for i in range(1, len(self.layers)):
            print(i)
            layer = LinearLayer(
                self.layers[i-1], 
                self.layers[i], 
                activation_fn_class=self.activation_fn_classes[i-1], 
                weight_init=self.weight_init[i-1],
                learning_rate=self.learning_rate,
                optimizer = self.optimizer
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
        training_loss_class = self.loss_fn_class(l2_lambda=1e-4) # turn l2 regularization off
        # performs a backward pass through the model
        y_hat = self.forward(x, backprop=True) # backprop true to save self.linear_transform and self.activation in each layer

        loss = training_loss_class.forward(y=y, y_hat=y_hat, weights=self.model[-1].weights) # pass the weights for the last layer
        loss_gradient = training_loss_class.derivative(y=y, y_hat=y_hat) # needs to be combined with softmax
        
        gradient = loss_gradient
        # start from the last layer of the model
        for layer in reversed(self.model):
            # print(layer)
            
            gradient = layer.backward(gradient) # relys on the layer backward function

        return loss

            



