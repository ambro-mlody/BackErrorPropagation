import numpy as np


class Neuron(object):

    def __init__(self, no_of_inputs, eta=0.1):
        self.no_of_inputs = no_of_inputs
        self.eta = eta
        self.weights = np.random.random(no_of_inputs + 1) - 0.5
        self.activation = 0
        self.delta = 0
        self.sigma = 0

    def sigmoid(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))

    def sigma_derivative(self, x):
        self.sigma = self.sigmoid(x)
        return self.sigma * (1 - self.sigma)

    def output(self, x):
        a = np.dot(self.weights[1:], x) + self.weights[0]
        self.sigma = self.sigmoid(a)
        return self.sigma

    def update_weights(self, x):
        for i in range(1, len(self.weights)):
            #print("W:", self.weights, "e:", self.eta, "d:", self.delta, "x", x)
            self.weights[i] -= self.eta * self.delta[i - 1] * x[i - 1]
        self.weights[0] -= self.eta * self.delta[0]
