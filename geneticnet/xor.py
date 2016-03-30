__author__ = 'wisienkas'
from numpy.random import random, binomial
import numpy

class Population:

    # Network structure
    #
    #   x1->w1  (u1) -> w5
    #     \ w2              (u3)
    #     / w3
    #   x2->w4  (u2) -> w6
    #
    def __init__(self, values):
        self.values = values
        self.w1 = values[0]
        self.w2 = values[1]
        self.w3 = values[2]
        self.w4 = values[3]
        self.u1 = values[4]
        self.u2 = values[5]
        self.w5 = values[6]
        self.w6 = values[7]
        self.u3 = values[8]

    def fitness(self, train, test):
        error = 0
        for i in range(0, 4):
            error += 0 if test[i] is self.predict(train[i]) else 1

        return error

    def predict(self, x):
        x1 = x[0]
        x2 = x[1]
        upper = 1 if self.w1 * x1 + self.w3 * x2 > self.u1 else 0
        lower = 1 if self.w2 * x1 + self.w4 * x2 > self.u2 else 0

        result = 1 if upper * self.w5 + lower * self.w6 > self.u3 else 0

        return result

pop_size = 50
mutate = 0.4
mask_size = 9
generations = 1000
tournament_size = 10
offset = 10
best = None

train = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

test = [0, 1, 1, 0]



def tupling(pop, train, test):
    return (pop, pop.fitness(train, test))


def mask():
    return numpy.array(binomial(1, 0.5, mask_size))


def generate():
    return Population([(x * (offset * 2)) - offset for x in random(mask_size)])


def mutating(arr):
    if binomial(1, mutate):
        arr[numpy.random.choice(mask_size, 1)] = randomValue()
    return arr

def new_pop(a, b):
    masker = mask()
    val1 = mutating([a.values[i] if masker[i] > 0.5 else b.values[i] for i in range(mask_size)])
    pop1 = Population(val1)

    val2 = mutating([a.values[i] if masker[i] > 0.5 else b.values[i] for i in range(mask_size)])
    pop2 = Population(val2)

    return [pop1, pop2]


def randomValue():
    return (random() * (2 * offset)) - offset


def tournament(generation, combatents):
    fighter_ids = numpy.random.choice(len(generation), combatents, replace=False)
    fighters = [generation[index] for index in fighter_ids]
    return sorted(fighters, key=lambda f: f[1])


def print_best():
    print("Best is found at generation: %s, error: %s, cost: %s, with values: %s" % (g, best[1], pop_size * (g + 1), best[0].values))

first_generation = [generate() for i in range(pop_size)]

#first_generation[0] = Population([1, -1, 1, -1, 0.5, -1.5, 1, 1, 1.5])
#first_generation[0].fitness(train, test)

import numpy.matlib

for g in range(generations):
    rated = sorted([tupling(p, train, test) for p in first_generation], key=lambda k: k[1])
    if best is None or best[1] > rated[0][1]: best = rated[0]

    if best is not None and best[1] == 0:
        print_best()
        break

    if best is not None and g % 1000 == 999:
        print_best()

    newgen = []
    for pair in range(int(pop_size / 2)):
        winners = tournament(rated, tournament_size)
        new_pair = new_pop(winners[0][0], winners[1][0])
        newgen.append(new_pair[0])
        newgen.append(new_pair[1])

    first_generation = newgen

print_best()