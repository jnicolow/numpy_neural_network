import numpy as np

class CategoricalCrossEntropy:
    """
    soft max is incorperated with this to allow computation of softmax gradient SO last layer of model can use DummyActivation (does nothing)

    """
    def __init__(self, l2_lambda=0.0):  # add L2 regularization coefficient
        self.l2_lambda = l2_lambda

    def forward(self, y, y_hat, weights=None):
        exp_values = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        softmax_output = np.clip(softmax_output, 1e-15, 1 - 1e-15)

        # cross-entropy 
        batch_size = y.shape[0]
        loss = -np.sum(y * np.log(softmax_output)) / batch_size
        
        if not weights is None:
            # L2 regularization term added to loss
            l2_reg = self.l2_lambda * np.sum(weights ** 2)
        else: l2_reg = 0
        return loss + l2_reg

    def derivative(self, y, y_hat):
        exp_values = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return (softmax_output - y) / y.shape[0]