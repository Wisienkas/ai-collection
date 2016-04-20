import math

__author__ = 'wisienkas'
import numpy as np
import scipy.stats as st

types = {'class',
         'regression',}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class neuralnet:

    def __init__(self, shape, type='class', learning=0.05):
        self.valid_type(type)
        self.learning = learning
        self.type = type
        self.shape = shape
        self.units = self.calculate_unit_shape(shape)
        self.weights = self.calculate_weight_shape(shape)
        self.forward_memory = self.calculate_unit_shape(shape)
        self.backward_memory = self.calculate_unit_shape(shape)

    def valid_type(self, type):
        if type not in types:
            raise Exception("Invalid type, has to be from: %s" % types)

    def calculate_unit_shape(self, shape):
        unit_list = [np.random.rand(layer) for layer in shape]
        return unit_list

    def calculate_weight_shape(self, shape):
        weight_list = [self.get_weight_layer(i,layer, shape) for i, layer in enumerate(shape)]
        return weight_list

    def get_weight_layer(self, i, layer, shape):
        if i is 0:
            return np.random.rand(layer, layer)
        else:
            return np.random.rand(shape[i -1], layer)

    def valid_data(self, data, expected = None):
        if expected is None:
            expected = self.shape[0]
        if len(data) is not expected:
            raise Exception("Invalid data len", data, self.shape)

    def forward(self, data):
        r = None
        for layer in range(len(self.shape)):
            if layer is len(self.shape) - 1 and self.type is 'class':
                return self.forward_layer_threshold(layer, input_data)
            input_data = self.forward_memory[layer - 1] if layer > 0 else data
            r = self.forward_layer(layer, input_data)

        print("gate_in: %s, units: %s, result: %s" % (self.forward_memory[-1], self.units[-1], r))
        return r

    def forward_layer(self, layer, data):
        for index, dest in enumerate(self.units[layer]):
            self.forward_memory[layer][index] = sigmoid(np.sum(data * self.weights[layer][::,index]) + dest)
        return self.forward_memory[layer]

    def forward_layer_threshold(self, layer, data):
        for index, dest in enumerate(self.units[layer]):
            val = np.sum(data * self.weights[layer][index]) - dest
            self.forward_memory[layer][index] = 1 if val >= 0 else 0

        return self.forward_memory[layer]

    def backward(self, input, expected):
        self.calculate_error(expected)
        self.update_units()
        self.update_weights(input)

    def update_units(self):
        for layer in range(len(self.shape)):
            delta_units = self.learning * self.backward_memory[layer]
            new_unit = self.units[layer] + delta_units
            old_unit = self.units[layer]
            self.units[layer] = new_unit

    def calculate_error(self, expected):
        for layer in reversed(range(len(self.shape))):
            out = self.forward_memory[layer]
            if layer >= len(self.shape) - 1:
                error = (expected - out)
                self.backward_memory[layer] = error
            else:
                error_sum = self.error(layer)
                error = out * (1 - out) * error_sum
                self.backward_memory[layer] = error

    def error(self, layer):
        err = np.multiply(self.backward_memory[layer + 1], self.weights[layer + 1])
        return np.sum(err, axis=1)

    def update_weights(self, input):
        for layer in reversed(range(len(self.shape))):

            # Values fed to layer
            input_layer = self.forward_memory[layer - 1] if layer > 0 else input
            # Error on output from layer
            out_layer = self.backward_memory[layer]
            if layer is len(self.shape) - 1:
                for in_key, in_val in enumerate(input_layer):
                    for out_key, out_val in enumerate(out_layer):
                        delta = self.learning * out_val * in_val
                        new_weight = self.weights[layer][in_key, out_key] + delta
                        self.weights[layer][in_key, out_key] = new_weight
            else:
                for in_key, in_val in enumerate(input_layer):
                    for out_key, out_val in enumerate(out_layer):
                        delta = self.learning * out_val * in_val
                        new_weight = self.weights[layer][in_key, out_key] + delta
                        self.weights[layer][in_key, out_key] = new_weight

    def train(self, data, expected):
        result = self.forward(data)
        self.backward(data, expected)
        return result

nn = neuralnet((2,8,2), learning=0.01)

for i in range(100):
    if (nn.train([3,3], [0,0]) == [0,0]).all():
        print("iterations: %s" % i)
        break
