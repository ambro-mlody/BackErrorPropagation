from layer import Layer
import numpy as np


class NeuralNetwork(object):

    def __init__(self, structure, iterations=100, eta=0.1):
        self.structure = structure
        self.iterations = iterations
        self.eta = eta
        self.layers = []
        self.output = []
        self.errors = []
        for i in range(1, len(structure)):
            self.layers.append(Layer(self.structure[i], self.structure[i - 1], eta=eta))

    def train(self, traindata_x, traindata_y):
        for i in range(self.iterations):
            for _ in range(len(traindata_x)):
                index = int(np.random.random() * len(traindata_x))
                x = traindata_x[index]
                y = traindata_y[index]

                yo = self.forward(x)
                self.backward(y)

            self.errors.append(self.error(traindata_x, traindata_y))

    def forward(self, x):
        self.layers[0].inputs = x.copy()
        inputs = self.layers[0].inputs
        for i in range(len(self.structure) - 2):
            inputs = self.layers[i].output(inputs)
            self.layers[i + 1].inputs = inputs

        self.output = self.layers[i + 1].output(inputs).copy()
        return self.output

    def backward(self, output):
        last_layer = len(self.layers) - 1
        input = self.layers[last_layer].inputs.copy()
        for j in range(self.layers[last_layer].layer_size):
            epsilon = output[j] - self.output[j]
            self.layers[last_layer].neurons[j].delta = epsilon * self.layers[last_layer].neurons[j].sigma_derivative(
                input)
        self.layers[last_layer].update_weights(input)

        for l in reversed(range(len(self.layers) - 1)):
            input = self.layers[l + 1].inputs.copy()
            for j in range(self.layers[l].layer_size):
                epsilon = 0
                for k in range(self.layers[l + 1].layer_size):
                    epsilon += self.layers[l + 1].neurons[k].weights[j] * self.layers[l + 1].neurons[k].delta

                self.layers[l].neurons[j].delta = epsilon * self.layers[l].neurons[j].sigma_derivative(input)
            self.layers[l].update_weights(input)

    def error(self, traindata_x, traindata_y):
        e = 0
        for (x, y) in zip(traindata_x, traindata_y):
            yo = self.forward(x)
            e += np.linalg.norm(y - yo) ** 2
        return e

