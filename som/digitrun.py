from os import system
import os

__author__ = 'wisienkas'
import numpy as np
from scipy import ndimage, misc
from sklearn.feature_extraction import image
from mnist import MNIST
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from som.SelfOrganizedMaps import Som


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

def plot_som(shape, h, w):
    gridspec = get_grid_subs(h, w)
    subs = [plt.subplot(grid) for grid in gridspec]
    for key, sub in enumerate(subs):
        i, j = iter[key]
        sub.im = sub.imshow(s.outputs[i,j,::].reshape((shape[0], -1)), cmap = plt.get_cmap('gray'))
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




print(os.getcwd())
mndata = MNIST('../mnist-digits')
train = mndata.load_training()

train_pixels = np.array(train[0]) / 255.0

h = 20
w = 20
s = Som(len(train[0][0]), hdim=h, wdim=w, learning_rate=0.05, sigma=1, a=50000, b=30000)
iter = index_map(h, w).reshape(h * w, 2)
#fig, subs = plot_som()

#anim = animation.FuncAnimation(fig, update, init_func=init, interval=10)
#misc.imshow(img_gray)
#plot_img_patches((pax,pay), h, w, patches)
#plt.show()

#plot_som((pax, pay), h, w)
#plt.show()
i = 0
for epoch in range(1):
    for pixels in train_pixels:
        i += 1
        if i == 20000:
            break
        s.train(pixels)

# Showing the actual Self Organized Map
# After Training
plot_som((28, 28), h, w)
plt.show()

# Showing the Self Organized Map U-matrix
goodness = s.get_goodness(3)
plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.show()

print("Finished")