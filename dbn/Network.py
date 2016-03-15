import math

__author__ = 'wisienkas'
from numpy.random import binomial as sampler
from numpy.random import random

learning_rate = 0.001

class NeuralNetwork:

    hidden_layers = []
    weights = {}

    def __init__(self, layers):
        if not isinstance(layers, [].__class__):
            raise Exception("Invalid layer input")
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer):
        predecessors = None
        if len(self.hidden_layers) is not 0:
            predecessors = self.hidden_layers[-1]

        current_layer = []
        for i in range(0, layer):
            node = Node()
            if predecessors is not None:
                for old_node in predecessors:
                    self.weights[(old_node, node)] = random()
            current_layer.append(node)

        self.hidden_layers.append(current_layer)

    def test_input(self, x):
        if len(x) is not len(self.hidden_layers[0]):
            raise Exception("Input len is %s, while required to be %s", (len(x), len(self.hidden_layers[0])))

    def train(self, x, layer):
        self.test_input(x)

        # Save all the input variables on the visible layer
        visible_layer = self.hidden_layers[0]
        for index, element in enumerate(x):
            visible_layer[index].last_result = element

        # Define current layer
        current_layer = 0

        # Feed forward to reach layer which has to be trained
        while layer is not current_layer + 1:
            self.forward(current_layer)
            current_layer += 1

        # Setup the from to relation ship for training
        predecessor = self.hidden_layers[layer - 1]
        successor = self.hidden_layers[layer]

        self.restricted_boltzman_machine(predecessor, successor)

    # The steps involved in restricted boltzman machine is following
    #
    # 1. Calculate H1 for all hidden nodes
    # 2. Calculate X2 (going backwards)
    # 3. Calculate H2
    # 4. Adjust bias and weights
    def restricted_boltzman_machine(self, predecessor, successor):
        # defining mappings for values
        h1_map = {}
        x2_map = {}
        h2_map = {}

        # Calculate H1
        for hidden_node in successor:
            # Sum up all the connections to this node from visible layer
            sum = 0
            for visible_node in predecessor:
                sum += visible_node.last_result * self.weight(visible_node, hidden_node)
            h1_map[(hidden_node)] = sampler(1, sigmoid(sum + hidden_node.bias))

        # Calculate X2
        for visible_node in predecessor:
            sum = 0
            for hidden_node in successor:
                sum += h1_map.get((hidden_node)) * self.weight(visible_node, hidden_node)
            x2_map[(visible_node)] = sigmoid(visible_node.bias + sum)

        # Calculate H2
        for hidden_node in successor:
            sum = 0
            for visible_node in predecessor:
                sum += x2_map.get((visible_node)) * self.weight(visible_node, hidden_node)
            h2_map[(hidden_node)] = sigmoid(sum + hidden_node.bias)

        # Adjust weights
        for hidden_node in successor:
            for visible_node in predecessor:
                x1 = visible_node.last_result
                x2 = x2_map.get((visible_node))
                h1 = h1_map.get((hidden_node))
                h2 = h2_map.get((hidden_node))
                # Weight delta
                weight_delta = learning_rate * (x1 * h1 - x2 * h2)
                self.weights[(visible_node, hidden_node)] += weight_delta
        # Adjust Hidden bias
        for hidden_node in successor:
            h1 = h1_map.get((hidden_node))
            h2 = h2_map.get((hidden_node))
            hidden_node.bias += learning_rate * (h1 - h2)
        # Adjust visible bias
        for visible_node in predecessor:
            x1 = visible_node.last_result
            x2 = x2_map.get((visible_node))
            visible_node.bias += learning_rate * (x1 - x2)

    def weight(self, a, b):
        return self.weights.get((a, b))

    # The forward function
    def forward(self, current_layer):
        layer_from = self.hidden_layers[current_layer]
        layer_to = self.hidden_layers[current_layer + 1]

        for out_node in layer_to:
            # Summing up all conncetions to this node "out_node"
            sum = 0
            for in_node in layer_from:
                sum += in_node.last_result * self.weights.get((in_node, out_node))
            # Use sigmoid to get the result and save it to the node
            out_node.last_result = sigmoid(out_node.bias + sum)

    def predict(self, x):
        self.test_input(x)

        for layer in range(0, len(self.hidden_layers) - 1):
            self.forward(layer)

        return [n.last_result for n in self.hidden_layers[-1]]


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