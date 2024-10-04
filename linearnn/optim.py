import numpy as np

class AdamOptimizer():
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None # first moment
        self.v = None # second moment
        self.t = 0 # timestep


    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.m = np.zeros_like(weights)