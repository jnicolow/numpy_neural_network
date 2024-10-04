import numpy as np

class AdamOptimizer:
    # this will be initialized seperately for weight and bias (and seperatly for each layer)
    # there should be a seperate self.m and v for each layer of the model (and for each weight/bias)
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None # first moment 
        self.v = None # second raw moment
        self.t = 0 # timestep


    def update(self, weights, gradients):
   
        if self.m is None: # this will spur the first initialization
            self.m = np.zeros_like(weights) # (initialized to zero in adam paper) https://arxiv.org/pdf/1412.6980
            self.v = np.zeros_like(weights)

        self.t += 1 # update time
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradients # update first moment
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (gradients ** 2) # update second moment

        m_hat = self.m / (1 - self.beta_1 ** self.t) # bias correction for first moment
        v_hat = self.v / (1 - self.beta_2 ** self.t) # bias correction for first moment

        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon) # weight update

        return weights
