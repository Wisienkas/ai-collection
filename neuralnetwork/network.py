import math

__author__ = 'wisienkas'
from numpy.random import random

learning_rate = 0.001

class NeuralNetwork:

    network_layers = []
    weights = {}

    def __init__(self, layers):
        if not isinstance(layers, [].__class__):
            raise Exception("Invalid layer input")
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        predecessors = None
        if len(self.network_layers) is not 0:
            predecessors = self.network_layers[-1]

        current_layer = []
        for i in range(0, layer):
            node = Node()
            if predecessors is not None:
                for old_node in predecessors:
                    self.weights[(old_node, node)] = random()
            current_layer.append(node)

        self.network_layers.append(current_layer)

    def test_input(self, x):
        if len(x) is not len(self.network_layers[0]):
            raise Exception("Input len is %s, while required to be %s", (len(x), len(self.network_layers[0])))

    def train(self, x):
        self.test_input(x)

        self.predict(x)
        self.adapt(x)

    def weight(self, a, b):
        return self.weights.get((a, b))

    # The forward function
    def forward(self, current_layer):
        layer_from = self.network_layers[current_layer]
        layer_to = self.network_layers[current_layer + 1]

        for out_node in layer_to:
            # Summing up all conncetions to this node "out_node"
            sum = 0
            for in_node in layer_from:
                sum += in_node.last_result * self.weights.get((in_node, out_node))
            # Use sigmoid to get the result and save it to the node
            out_node.last_result = sigmoid(out_node.bias + sum)

    def predict(self, x):
        self.test_input(x)
        for index, visible_node in enumerate(self.network_layers[0]):
            visible_node.last_result = x[index]

        for layer in range(0, len(self.network_layers) - 1):
            self.forward(layer)

        return [n.last_result for n in self.network_layers[-1]]


class Node:

    last_result = None

    def __init__(self, id = None, bias = None):
        if id is None:
            self.id = random()
        else:
            self.id = id

        if bias is None:
            self.bias = random()
        else:
            self.bias = bias


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


neu = NeuralNetwork([2, 2])

def print_info(neu):
    for n1 in neu.hidden_layers[0]:
        print("bias: %s, id: %s" % (n1.bias, n1.id))
        for n2 in neu.hidden_layers[1]:
            print("Connected to id: %s, weight: %s" % (n2.id, neu.weights.get((n1, n2))))

print_info(neu)

training = [[random(), random()] for x in range(0, 10000)]

for x in training: neu.train(x, 1)
print("\n##################################################\n")

print_info(neu)