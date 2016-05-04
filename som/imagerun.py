__author__ = 'wisienkas'
import numpy as np
from scipy import ndimage, misc
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from som.SelfOrganizedMaps import Som


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def block3d(a, nrow, ncols):
    M, N, r = a.shape
    return a.reshape(M/nrow,nrow,N/ncols,ncols,r).transpose(0,2,1,3,4).reshape(-1,nrow,ncols,r)


def block(arr, nrows, ncols):
    return arr.reshape(arr.shape[0]/nrows, ncols, arr.shape[1]/nrows, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)


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
    subs = get_grid_subs(h, w)
    for key, sub in enumerate(subs):
        i, j = iter[key]
        dat = s.outputs[i,j,::].reshape(shape)
        sub.im = sub.imshow(dat)
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


DEBUG = False
epocs = 100
# img = misc.mread('img/image.jpg') / 255
img = misc.imread('img/face.png') / 255
# to gray using G = R*0.299 + G*0.587 + B*0.114
img_gray = np.sum(img * [0.299, 0.587, 0.144], axis=2)
x, y = img_gray.shape
px, py = (20, 20)
#patches = blockshaped(img_gray, 20, 20)
#patches = block(img_gray, px, py)
patches = block3d(img, px, py)
#patches = img_gray.reshape((20, 20, 10, 10))


l = len(img_gray.flatten())
#iseven = sum(img_gray.flatten() == patches.flatten()) == l
#patches = image.extract_patches_2d(img_gray, (px, py))
pac, pax, pay, paz = patches.shape

h = x / px
w = y / py
s = Som(pax * pay * paz, hdim=h, wdim=w, learning_rate=0.15, sigma=0.8)

# Will set image as weights on som
#s.outputs = patches.reshape((h, w, -1))
#print(s.outputs.shape)

iter = index_map(h, w).reshape(h * w, 2)
#fig, subs = plot_som()

#anim = animation.FuncAnimation(fig, update, init_func=init, interval=10)
#misc.imshow(img_gray)

if(DEBUG):
    plot_img_patches((pax,pay, 3), h, w, patches)
    plt.show()

    plot_som((pax, pay, 3), h, w)
    plt.show()

#for i, k in enumerate(iter):
#    assert sum(patches[i].flatten('C') == s.outputs[k[0], k[1]]) == pax * pay * paz

print("Will run %s iterations now" % pac * epocs)
for epoch in range(epocs):
    for patch in patches:
        s.train(patch.flatten('C'))

plot_som((pax, pay, 3), h, w)
plt.show()

goodness = s.get_goodness(3)
plt.imshow(goodness, cmap=plt.get_cmap('gray'), interpolation="nearest")
plt.show()

print("Finished")