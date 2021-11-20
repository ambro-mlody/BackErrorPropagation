from neuron import Neuron
import numpy as np


class Layer(object):
    def __init__(self, layer_size, prev_layer_size, eta=0.1):
        self.layer_size = layer_size
        self.prev_layer_size = prev_layer_size
        self.eta = eta
        self.neurons = []
        self.inputs = []
        for i in range(layer_size):
            self.neurons.append(Neuron(prev_layer_size, eta=self.eta))

    def output(self, x):
        result = []
        for i in range(self.layer_size):
            result.append(self.neurons[i].output(x))
        return np.array(result)

    def update_weights(self, x):
        for i in range(self.layer_size):
            self.neurons[i].update_weights(x)
