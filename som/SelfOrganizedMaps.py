import numpy as np
import math
import scipy.stats as stats
from scipy.spatial.distance import euclidean

__author__ = 'wisienkas'


# Will generate a distance map from the point
# to the indexes
# Used for neighbourhood function
def index_map_distmap(x, y, point):
    coords = np.empty((x, y, 2), dtype=np.intp)
    coords[..., 0] = np.arange(x)[:, None]
    coords[..., 1] = np.arange(y)

    return np.swapaxes(np.hypot(*(coords - point).T), 0, 1)


# Will return a normalized distance map
# The max value would be from 1 corner to the opposite
def index_map_distmap_normalized(x, y, point):
    index_map = index_map_distmap(x, y, point)
    return index_map / euclidean([0,0], [x,y])


# Will find the distance from the point
# to the point given for each x,y coord in the matrix
def distance_map(matrix, point):
    dist = (matrix - point)**2
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)

    return dist

#    return np.hypot(*(matrix - point).T)
#    return np.swapaxes(np.hypot(*(matrix - point).T), 0, 1)


def learning_rate_function(a_0, a_T, t, T):
    # Defines the learning using the "Power Series"
    # Function defined as:
    #   a_t = a_0 ( a_0 / a_T ) ^ (t / T)
    #return a_0 * (( a_0 / a_T) ** (t / T))
    # Inverse time
    # Function defined as:
    #   a_t = a / (t + b)
    return 10 / (t + 100)


class Som:

    def __init__(self, inputdim, hdim, wdim, learning_rate=0.05):
        self.learning = learning_rate
        self.iteration = 0
        self.hdim = hdim
        self.wdim = wdim
        self.inputdim = inputdim
        self.outputs = np.random.rand(hdim, wdim, inputdim)

    def train(self, data):
        self.validate(data)
        self.iteration += 1
        kernel = stats.norm()
        learning_rate = learning_rate_function(self.learning, 0.001, self.iteration, 1000)
        max_std = 3

        diff_matrix = distance_map(self.outputs, data)
        best_point = np.unravel_index(diff_matrix.argmin(), diff_matrix.shape)

        dist_matrix = index_map_distmap_normalized(self.hdim, self.wdim, best_point)
        kernel_matrix = kernel.pdf(dist_matrix * max_std)
        diff = learning_rate * (data - self.outputs)
        # Now to apply kernel matrix, we roll the array to fit shape
        # goes from shape (10x, 10y, 3z) -> (3z, 10x, 10y)
        diff = np.rollaxis(diff, 2) * kernel_matrix
        # Now invert the roll, to get back in original position (3z, 10x, 10y) -> (10x, 10y, 3z)
        diff = np.rollaxis(diff, 0, 3)
        # Update weights
        self.outputs += diff
        print("learning %s" % learning_rate)

    def validate(self, data):
        if not isinstance(data, [].__class__) or len(data) is not self.inputdim:
            raise Exception("Data is not list or Data and inputdim not same length!!! data: %s, inputdim: %s" % (len(data), self.inputdim))