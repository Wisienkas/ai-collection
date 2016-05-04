import numpy as np
import math
import scipy.stats as stats
from scipy.spatial.distance import euclidean, mahalanobis, correlation

__author__ = 'wisienkas'


# Will generate a distance map from the point
# to the indexes
# Used for neighbourhood function
def index_map_distmap(x, y, point):
    coords = index_map(x, y)
    return np.swapaxes(np.hypot(*(coords - point).T), 0, 1)


def index_map(x, y):
    coords = np.empty((x, y, 2), dtype=np.intp)
    coords[..., 0] = np.arange(x)[:, None]
    coords[..., 1] = np.arange(y)

    return coords

# Will return a normalized distance map
# The max value would be from 1 corner to the opposite
def index_map_distmap_normalized(x, y, point):
    index_map = index_map_distmap(x, y, point)
    return index_map / euclidean([0,0], [x,y])


def get_direction_score(direction_map, coord, matrix):
    sum = 0
    x, y = matrix.shape[:2]
    a, b = coord
    point = matrix[a, b, ::]
    n = 0
    for i, k in enumerate(direction_map):
        if k[0] is 0 and k[1] is 0:
            continue
        if 0 <= a + k[0] < x and 0 <= b + k[1] < y:
            neighbour = matrix[a + k[0], b + k[1],::]
            corr = correlation(point, neighbour)
            # It has a range of 0..2 and should be 0..1, so we divide by 2
            # See http://stackoverflow.com/questions/35988933/scipy-distance-correlation-is-higher-than-1
            sum += corr / 2
            n += 1

    return 1 - (sum / n)


def correlation_map(matrix, radius=5):
    x, y = matrix.shape[:2]
    imap = index_map(x, y).reshape(x * y, 2)
    radius_corr = (radius - 1) / 2
    direction_map = index_map(radius,radius).reshape(radius**2, 2) - [radius_corr, radius_corr]
    return np.asmatrix([get_direction_score(direction_map, k, matrix) for i, k in enumerate(imap)]).reshape(x, y)


# Will find the distance from the point using euclidian distance
# to the point given for each x,y coord in the matrix
def distance_map(matrix, point):
    dist = (matrix - point)**2
    dist = np.sum(dist, axis=2)
    dist = np.sqrt(dist)

    return dist

#    return np.hypot(*(matrix - point).T)
#    return np.swapaxes(np.hypot(*(matrix - point).T), 0, 1)


def learning_rate_function(a_0, a_T, t, T, a, b):
    # Defines the learning using the "Power Series"
    # Function defined as:
    #   a_t = a_0 ( a_0 / a_T ) ^ (t / T)
    #return a_0 * (( a_0 / a_T) ** (t / T))
    # Inverse time
    # Function defined as:
    #   a_t = a / (t + b)
    return a_0 * (a / (t + b))


class Som:

    def __init__(self, inputdim, hdim, wdim, learning_rate=0.05, sigma=3, a=2000, b=2000):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.learning = learning_rate
        self.iteration = 0
        self.hdim = hdim
        self.wdim = wdim
        self.inputdim = inputdim
        self.outputs = np.random.rand(hdim, wdim, inputdim)


    def train(self, data):
        #self.validate(data)
        self.iteration += 1
        learning_rate = learning_rate_function(self.learning, 0.001, self.iteration, 1000, a=self.a, b=self.b)
        kernel = stats.norm(0, self.sigma * learning_rate)

        diff_matrix = distance_map(self.outputs, data)
        best_point = np.unravel_index(diff_matrix.argmin(), diff_matrix.shape)

        dist_matrix = index_map_distmap_normalized(self.hdim, self.wdim, best_point)
        kernel_matrix = np.sqrt(kernel.pdf(dist_matrix))
        diff = learning_rate * (data - self.outputs)
        # Now to apply kernel matrix, we roll the array to fit shape
        # goes from shape (10x, 10y, 3z) -> (3z, 10x, 10y)
        diff = np.rollaxis(diff, 2) * kernel_matrix
        # Now invert the roll, to get back in original position (3z, 10x, 10y) -> (10x, 10y, 3z)
        diff = np.rollaxis(diff, 0, 3)
        # Update weights
        self.outputs += diff
        if self.iteration % 500 == 1:
            print("iteration: %s, learning %s, kernel %s, %s" % (self.iteration, learning_rate, np.amax(kernel_matrix), np.amin(kernel_matrix)))
            print("Max change: %s, Min change: %s" % (np.amax(diff), np.amin(diff)))

    def validate(self, data):
        if len(data) is not self.inputdim:
            raise Exception("Data is not list or Data and inputdim not same length!!! data: %s, inputdim: %s" % (len(data), self.inputdim))

    def get_goodness(self, radius = 5):
        matrix = self.outputs.copy()# * 2 - 1
        return correlation_map(matrix, radius=radius)

    def append_result_to_file(self, file):
        fd = open(file, 'a')
        fd.write()