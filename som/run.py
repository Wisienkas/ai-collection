import numpy as np
import matplotlib as mlp
import time
import matplotlib.pyplot as plt
from matplotlib import animation

from som.SelfOrganizedMaps import Som


__author__ = 'wisienkas'
colors = {'red':  [1.0, 0.0, 0.0],
         'green': [0.0, 1.0, 0.0],
         'blue':  [0.0, 0.0, 1.0]}




h = 9
w = 10
s = Som(3, hdim=h, wdim=w, learning_rate= 0.10)
#for h_i in range(h):
#    for w_i in range(w):
#        color = colors[numpy.random.choice(len(colors), 1)]
#        s.outputs[h_i, w_i, :] = color

fig = plt.figure()
im = plt.imshow(s.outputs, interpolation="nearest")

colorkeys = list(colors.keys())

def init():
    im.set_data(s.outputs)
    return [im]

def animate(i):
    key = np.random.choice(colorkeys)
    s.train(colors.get(key))

    # ma = np.amax(s.outputs)
    # mi = np.amin(s.outputs)

    # out = (s.outputs + np.random.rand(h, w, 3)) % 1

    # im.set_array((out - mi) / (ma - mi))
    im.set_array(s.outputs)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=100)

plt.show()

print("Finished")