import numpy as np

class ActivationFunction:
    def forward(self, x):
        raise NotImplementedError('forward not implemented for activation function')
    
    def derivative(self, x):
        raise NotImplementedError('derivative not implemented for activation function')
    

class DummyActivation(ActivationFunction):
    def forward(self, x):
        # simply return the input without any transformation
        return x
    
    def derivative(self, x):
        # the derivative of a linear function (i.e., f(x) = x) is just 1
        return np.ones_like(x)


class ReLU(ActivationFunction):
    # rectified linear unit
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    

class SELU(ActivationFunction):
    # scaled exponential linear unit
    def forward(self, x):
        x = np.clip(x, -500, 500)  # to prevent overflow in exp(x)
        return np.where(x <= 0, 1.75809 * (np.exp(x)-1), 1.0507 * x)
    
    def derivative(self, x):
        x = np.clip(x, -500, 500)  # to prevent overflow in exp(x)
        return np.where(x <= 0, 1.0507, 1.0507 * np.exp(x))
    

class Tanh(ActivationFunction):
    def forward(self, x):
        x = np.clip(x, -500, 500)  # prevent overflow
        exp_x = np.exp(x)
        exp_neg_x = np.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    
    def derivative(self, x):
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2
    


class Softmax(ActivationFunction):
    def forward(self, x):
        denominator = np.sum(np.exp(x)) # sum of all the e^logit for every logit
        return np.exp(x) / denominator # normalize e^logit value by sum of e^logits
    
    def derivative(self, output_gradient, y):
        # Done differentiate softmax, combine gradient with cross-ent loss
        return output_gradient - y