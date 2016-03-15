__author__ = 'wisienkas'
from matplotlib import pyplot
from random import *

step_size = 0.0002


class Unit:

    def __init__(self):
        self.w = (random() * 2) - 1


class Perceptron:

    def __init__(self, inputs):
        self.threshold = Unit()
        self.units = []
        for i in range(0, inputs):
            self.units.append(Unit())

    def learn(self, inputs, result):
        if len(inputs) != len(self.units):
            raise Exception("Invalid Input")

        predicted = self.predict(inputs)
        self.adapt(predicted, result, inputs)

    def predict(self, inputs):
        result = 0
        for i, unit_i in enumerate(self.units):
            result += unit_i.w * inputs[i]

        return result

    def adapt(self, predicted, result, inputs):
        for i, unit_i in enumerate(self.units):
            delta_w = step_size * (result - predicted) * inputs[i]
            unit_i.w += delta_w

perceptron = Perceptron(2)

vary = range(-10, 10)
inputs = sample(range(0,100), 100)
targets = []
for i in inputs:
    targets.append(i * 3 + randrange(-30, 30))

def evaluate():
    error = 0
    for i in range(0, 100):
        error += (targets[i] - perceptron.predict([inputs[i], 1]))**2
    error = error / 2
    print("error: {}".format(error))

pyplot.plot(inputs, targets, 'ro')
pyplot.plot([0, 100], [perceptron.predict([0, 1]), perceptron.predict([100, 1])], 'k-')
pyplot.show()

for epoch in range(0, 200):
    if epoch % 50 == 0: evaluate()
    for i in range(0, 100):
        perceptron.learn([inputs[i], 1], targets[i])

pyplot.plot(inputs, targets, 'ro')
pyplot.plot([0, 100], [perceptron.predict([0, 1]), perceptron.predict([100, 1])], 'k-')
pyplot.show()