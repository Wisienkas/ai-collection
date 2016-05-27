from os import system
import os

__author__ = 'wisienkas'
import numpy as np
from scipy import ndimage, misc
from sklearn.feature_extraction import image
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import som.SelfOrganizedMaps as som


def index_map(x, y):
    coords = np.empty((x, y, 2), dtype=np.intp)
    coords[..., 0] = np.arange(x)[:, None]
    coords[..., 1] = np.arange(y)

    return coords


def get_grid_subs(h, w, big = False):
    if big:
        gridspec = gs.GridSpec(int(h), int(w), left=0.001,bottom=0.001,right=0.999,top=0.999,wspace=0.01,hspace=0.01)
    else:
        gridspec = gs.GridSpec(int(h), int(w), left=0.001,bottom=0.001,right=0.999,top=0.93,wspace=0.01,hspace=0.01)
    return [plt.subplot(grid) for grid in gridspec]


def plot_img_patches(shape, h, w, data, color=True):
    subs = get_grid_subs(h, w)
    for key, sub in enumerate(subs):
        if color:
            sub.im = sub.imshow(data[key,::])
        else:
            sub.im = sub.imshow(data[key,::], cmap=plt.get_cmap("gray"))
        sub.set_xticks([])
        sub.set_yticks([])


def plot_som(som, shape, h, w, big = False):
    gridspec = get_grid_subs(h, w, big)
    subs = [plt.subplot(grid) for grid in gridspec]
    for key, sub in enumerate(subs):
        i, j = (sub.rowNum, sub.colNum)
        sub.im = sub.imshow(som.outputs[i,j,::].reshape((shape[0], -1)), cmap = plt.get_cmap('gray'))
        sub.set_xticks([])
        sub.set_yticks([])

#
# def init():
#     for key, sub in enumerate(subs):
#         i, j = iter[key]
#         sub.im.setdata(s.outputs[i,j,::].reshape((px, py), order='F'))
#
#
# def update(i):
#     for i in np.random.choice(len(patches), size=1000, replace=False):
#         s.train(patches[i])
#
#     for key, sub in enumerate(subs):
#         i, j = iter[key]
#         sub.im.setdata(s.outputs[i,j,::].reshape((px, py), order='F'))


def make_som_report(path, h, w, learning_rate, sigma, a, b, epochs,
                    max_iterations, goodness_range, prefix, pre_gradient, big=False):
    print("Started processing %s" % prefix)
    mndata = MNIST(path=path)
    train = mndata.load_training()

    train_pixels = np.array(train[0]) / 255.0

    s = som.Som(len(train[0][0]), hdim=h, wdim=w, learning_rate=learning_rate, sigma=sigma, a=a, b=b)

    if pre_gradient:
        outputs = som.index_map_distmap_normalized(h, w, np.array([0,0])) * np.ones((s.inputdim, h ,w), dtype=np.float)
        # position (z, 10x, 10y) -> (10x, 10y, z)
        s.outputs = np.rollaxis(outputs, 0, 3)

    # plot_som(s, (28, 28), h, w)
    # plt.show()
    # plt.clf()
    #
    # return

    goodness_list = list()
    goodness_ite = list()
    convergence = list()
    i = 0
    if max_iterations is None:
        for epoch in range(epochs):
            for pixels in train_pixels:
                convergence.append(s.train(pixels))
                if i % 100 is 0:
                    goodness = s.get_goodness(goodness_range)
                    absolute_goodness = np.absolute(goodness)
                    avg_fit = absolute_goodness.sum() / absolute_goodness.size
                    goodness_list.append(avg_fit)
                    goodness_ite.append(i)
                i += 1
        # Showing the actual Self Organized Map
        # After Training
        plot_som(s, (28, 28), h, w)
        plt.subplot("Map at Iteration: %s" % i)
        plt.savefig("%s-map.png" % prefix, dpi = 96 * 3)
        plt.clf()

        # Showing the Self Organized Map U-matrix
        goodness = s.get_goodness(goodness_range)

        plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="nearest")
        plt.savefig("%s-goodness.png" % prefix)
        plt.clf()
    else:
        breaker = max(max_iterations)
        while i <= breaker:
            for pixels in train_pixels:
                if i % 20 is 0:
                    goodness = s.get_goodness(goodness_range)
                    absolute_goodness = np.absolute(goodness)
                    avg_fit = absolute_goodness.sum() / absolute_goodness.size
                    goodness_list.append(avg_fit)
                    goodness_ite.append(i)
                for breaks in max_iterations:
                    if breaks == i:
                        if big:
                            print("At %s" % i)
                        # Showing the actual Self Organized Map
                        # After Training
                        plot_som(s, (28, 28), h, w)
                        if big:
                            plt.savefig("%s-mapAt%s.png" % (prefix, breaks), dpi = 96 * 4)
                        else:
                            plt.suptitle("Map At Iteration: %s" % i)
                            plt.savefig("%s-mapAt%s.png" % (prefix, breaks))
                        plt.clf()

                        # Showing the Self Organized Map U-matrix
                        goodness = s.get_goodness(goodness_range)

                        plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="gaussian")
                        if big:
                            plt.savefig("%s-goodnessAt%s.png" % (prefix, breaks), dpi = 96 * 4)
                        else:
                            plt.savefig("%s-goodnessAt%s.png" % (prefix, breaks))
                        plt.clf()

                i += 1
                if i > breaker:
                    break
                convergence.append(s.train(pixels))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    line1 = ax1.plot(goodness_ite, goodness_list, 'b-', label="Fit")
    line2 = ax2.plot(convergence, 'r-', label="convergence")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("fitting", color = 'b')
    ax2.set_ylabel('Convergence', color='r')

    for t1 in ax1.get_yticklabels():
        t1.set_color('b')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc = 7)
    plt.title("Map Progression Overview")
    plt.savefig("%s-graph.png" % prefix)
    plt.clf()
    plt.close()
    print("Finished %s" % prefix)

    return s

make_som_report("../mnist-digits", 10, 10, learning_rate=0.6, sigma=0.7, a=30, b=30, epochs=None,
                max_iterations=[50, 100, 200, 350, 500, 1000], goodness_range=3, prefix="img/experiment12-pre", pre_gradient=False)
# make_som_report("../mnist-digits", 10, 10, learning_rate=0.05, sigma=0.2, a=50, b=10, epochs=None,
#               max_iterations=[50, 200, 500], goodness_range=3, prefix="img/tests/experiment12-non", pre_gradient=False)
