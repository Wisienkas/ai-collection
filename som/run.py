import numpy as np
import matplotlib as mlp
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gs

from som.SelfOrganizedMaps import Som


__author__ = 'wisienkas'
# Put colors here in decimal format
colors = {'darkyellow': [0.5, 0.5, 0],
          'black': [0,0,0],
          'red': [1,0,0],
          'blue': [0,0,1],
          'yellow': [1,1,0],
          'orange': [1, 0.5,0],
          'green': [0,1,0]}


h = 20
w = 20
s = Som(3, hdim=h, wdim=w, learning_rate=0.10, sigma=1)
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

#anim = animation.FuncAnimation(fig, animate, init_func=init, interval=10)

#plt.show()


def get_grid_subs(h, w):
    gridspec = gs.GridSpec(int(h), int(w), left=0.001,bottom=0.001,right=0.999,top=0.999,wspace=0.01,hspace=0.01)
    return [plt.subplot(grid) for grid in gridspec]


def set_img_show(sub, data, color=True):
    if color:
        sub.imshow(data)
    else:
        sub.imshow(data, cmap=plt.get_cmap('gray'))
    sub.set_xticks([])
    sub.set_yticks([])


initial_values = s.outputs.copy()
initial_goodness = s.get_goodness()

for i in range(200):
    key = np.random.choice(colorkeys)
    s.train(colors.get(key))

final_value = s.outputs.copy()
final_goodness = s.get_goodness()


subs = get_grid_subs(2,2)
set_img_show(subs[0], initial_values)
set_img_show(subs[1], initial_goodness, color=False)
set_img_show(subs[2], final_value)
set_img_show(subs[3], final_goodness, color=False)

plt.show()

print("Finished")