import numpy
import math
import scipy.stats as stats

__author__ = 'wisienkas'

norm_0 = stats.norm()

class Som:

    def __init__(self, inputdim, hdim, wdim):
        self.iteration = 0
        self.hdim = hdim
        self.wdim = wdim
        self.inputdim = inputdim
        self.outputs = numpy.random.rand(hdim, wdim, inputdim)
        self.learning = 10.0

    def train(self, data):
        self.validate(data)
        self.iteration += 1
        hdim_min = None
        wdim_min = None
        val_min = None
        val = []
        for hdim in range(self.hdim):
            for wdim in range(self.wdim):
                val = []
                for inputdim in range(self.inputdim):
                    val.append(data[inputdim] * self.outputs[hdim, wdim, inputdim])
                if val_min is None or sum(val_min) > sum(val):
                    val_min = val
                    hdim_min = hdim
                    wdim_min = wdim

        for inputdim in range(self.inputdim):
            self.outputs[hdim_min, wdim_min, inputdim] += (self.learning / self.iteration) * (data[inputdim] - val[inputdim])

        for hdim in range(self.hdim):
            for wdim in range(self.wdim):
                dist = math.sqrt((hdim - hdim_min)**2 + (wdim - wdim_min)**2) / 20
                for inputdim in range(self.inputdim):
                    self.outputs[hdim, wdim, inputdim] += (self.learning) * (norm_0.pdf(dist)) * (data[inputdim] - self.outputs[hdim, wdim, inputdim])

    def validate(self, data):
        if not isinstance(data, [].__class__) or len(data) is not self.inputdim:
            raise Exception("Data is not list or Data and inputdim not same length!!! data: %s, inputdim: %s" % (len(data), self.inputdim))