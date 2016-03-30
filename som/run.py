import numpy
import matplotlib as mlp
import time
import matplotlib.pyplot as plt
from matplotlib import animation

from som.SelfOrganizedMaps import Som


__author__ = 'wisienkas'
colors = {'red':  [1.0, 0.0, 0.0],
         'green': [0.0, 1.0, 0.0],
         'blue':  [0.0, 0.0, 1.0]}




h = 10
w = 10
s = Som(3, hdim=h, wdim=w)
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
    key = numpy.random.choice(colorkeys)
    s.train(colors.get(key))

    im.set_array(s.outputs)
    return [im]

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1)

plt.show()

print("Finished")