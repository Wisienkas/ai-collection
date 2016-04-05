import math

__author__ = 'wisienkas'
import numpy as np

types = {'class',
         'regression',}


class neuralnet:


    def __init__(self, shape, type='class'):
        self.valid_type(type)
        self.type = type
        self.shape = shape
        self.units = self.calculate_unit_shape(shape)
        self.weights = self.calculate_weight_shape(shape)
        self.forward_memory = self.units.copy()
        self.backward_memory = self.units.copy()


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
            return np.random.rand(layer)
        else:
            return np.random.rand(shape[i -1], layer)

    def valid_data(self, data, expected = None):
        if expected is None:
            expected = self.shape[0]
        if len(data) is not expected:
            raise Exception("Invalid data len", data, self.shape)

    def forward(self, data):
        for layer in range(len(self.shape)):
            if layer is 0:
                self.forward_memory[layer] = self.weights[layer] * data + self.units[layer]
            elif layer is len(self.shape) - 1 and self.type is 'class':
                self.forward_memory[layer] = self.forward_layer_threshold(layer, input_data)
            else:
                input_data = self.forward_memory[layer - 1]
                self.forward_memory[layer] = self.forward_layer(layer, input_data)

        return self.forward_memory[-1]

    def forward_layer(self, layer, data):
        for index, dest in enumerate(self.units[layer]):
            self.forward_memory[layer][index] = np.sum(data * self.weights[layer][index]) + dest
        return self.forward_memory[layer]

    def forward_layer_threshold(self, layer, data):
        for index, dest in enumerate(self.units[layer]):
            val = np.sum(data * self.weights[layer][index])
            self.forward_memory[layer][index] = 1 if val >= dest else 0
        return self.forward_memory[layer]

    def backward(self, input, expected):
        self.calculate_error(expected)
        self.update_weights()

    def calculate_error(self, expected):
        for layer in reversed(range(len(self.shape))):
            out = self.forward_memory[layer]
            if layer >= len(self.shape) - 1:
                self.backward_memory[layer] = (expected - out)
            else:
                error_sum = self.error(layer)
                self.backward_memory[layer] = out * (1 - out) * (error_sum)

    def error(self, layer):
        err = self.backward_memory[layer + 1] * self.weights[layer + 1]
        return err

    def update_weights(self):
        return



nn = neuralnet((2,2,2))
output = nn.forward([3,3])
nn.backward([3,3], [0,0])
print("lol")
