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


def get_grid_subs(h, w):
    gridspec = gs.GridSpec(int(h), int(w), left=0.001,bottom=0.001,right=0.999,top=0.999,wspace=0.01,hspace=0.01)
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

def plot_som(som, shape, h, w):
    gridspec = get_grid_subs(h, w)
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


def make_som_report(path, h, w, learning_rate, sigma, a, b, epochs, max_iterations, goodness_range, prefix, pre_gradient):
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

    if max_iterations is None:
        for epoch in range(epochs):
            for pixels in train_pixels:
                s.train(pixels)
    else:
        i = 0
        while i < max_iterations:
            for pixels in train_pixels:
                i += 1
                if i >= max_iterations:
                    break
                s.train(pixels)

    # Showing the actual Self Organized Map
    # After Training
    plot_som(s, (28, 28), h, w)
    plt.savefig("%s-map.png" % prefix)
    plt.clf()

    # Showing the Self Organized Map U-matrix
    goodness = s.get_goodness(goodness_range)

    plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="nearest")
    plt.savefig("%s-goodness.png" % prefix)

    plt.close()
    print("Finished")


for learning_rate in [0.01, 0.05, 0.15]:
    for sigma in [0.5, 1, 2]:
        prefix = "img/noGradient-learning_rate(%s)-sigma(%s)" % (learning_rate, sigma)
        make_som_report("../mnist-digits", 10, 10, learning_rate=learning_rate, sigma=sigma, a=50000, b=5000, epochs=2,
                max_iterations=None, goodness_range=3, prefix=prefix, pre_gradient=False)
        prefix = "img/gradient-learning_rate(%s)-sigma(%s)" % (learning_rate, sigma)
        make_som_report("../mnist-digits", 10, 10, learning_rate=learning_rate, sigma=sigma, a=50000, b=5000, epochs=2,
                max_iterations=None, goodness_range=3, prefix=prefix, pre_gradient=True)


# mndata = MNIST('../mnist-digits')
# train = mndata.load_training()
#
# train_pixels = np.array(train[0]) / 255.0
#
# h = 50
# w = 50
# s = Som(len(train[0][0]), hdim=h, wdim=w, learning_rate=0.05, sigma=1, a=50000, b=30000)
# iter = index_map(h, w).reshape(h * w, 2)
# #fig, subs = plot_som()
#
# #anim = animation.FuncAnimation(fig, update, init_func=init, interval=10)
# #misc.imshow(img_gray)
# #plot_img_patches((pax,pay), h, w, patches)
# #plt.show()
#
# #plot_som((pax, pay), h, w)
# #plt.show()
# i = 0
# for epoch in range(1):
#     for pixels in train_pixels:
#         i += 1
#         if i == 2000:
#             break
#         s.train(pixels)
#
# # Showing the actual Self Organized Map
# # After Training
# plot_som((28, 28), h, w)
# plt.savefig('dig.png')
# plt.show()
#
# # Showing the Self Organized Map U-matrix
# goodness = s.get_goodness(3)
# plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="nearest")
# plt.show()
#
# print("Finished")