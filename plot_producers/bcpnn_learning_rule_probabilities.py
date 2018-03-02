import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['figure.figsize'] = (16, 12)

np.set_printoptions(suppress=True, precision=2)

sns.set(font_scale=2.5)

def log_epsilon(x, epsilon=1e-10):

    return np.log(np.maximum(x, epsilon))

num = 100

p_i = 1.0
p_j_vector = np.linspace(0.1, 1.0, num=num)
p_ij_vector = np.linspace(0.1, 1.0, num=num)

w1 = np.zeros((num, num))

for index_x, p_j in enumerate(p_j_vector):
    for index_y, p_ij in enumerate(p_ij_vector):
        w1[index_y, index_x] = log_epsilon(p_ij / (p_i * p_j))




p_i = 0.5
p_j_vector = np.linspace(0.1, 1.0, num=num)
p_ij_vector = np.linspace(0.1, 1.0, num=num)

w2 = np.zeros((num, num))

for index_x, p_j in enumerate(p_j_vector):
    for index_y, p_ij in enumerate(p_ij_vector):
        w2[index_y, index_x] = log_epsilon(p_ij / (p_i * p_j))



p_i = 0.1
p_j_vector = np.linspace(0.1, 1.0, num=num)
p_ij_vector = np.linspace(0.1, 1.0, num=num)

w3 = np.zeros((num, num))

for index_x, p_j in enumerate(p_j_vector):
    for index_y, p_ij in enumerate(p_ij_vector):
        w3[index_y, index_x] = log_epsilon(p_ij / (p_i * p_j))

######
# Plot everything
#######

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

vmax = np.max([w1, w2, w3])
vmin = np.min([w1, w2, w3])

extent = [p_j_vector[0], p_j_vector[-1], p_ij_vector[0], p_ij_vector[-1]]

cmap = 'coolwarm'
im1 = ax1.imshow(w1, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax1.grid()
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
ax1.set_title(r'$p_i = 1.0$')
ax1.set_xlabel(r'$p_j$')
ax1.set_ylabel(r'$p_{ij}$')

im2 = ax2.imshow(w2, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax2.grid()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.set_title(r'$p_i = 0.5$')
ax2.set_xlabel(r'$p_j$')

im3 = ax3.imshow(w3, origin='lower', cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
ax3.grid()
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical');
ax3.set_title(r'$p_i = 0.1$')
ax3.set_xlabel(r'$p_j$')

plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

fig.savefig('./plot_producers/bcpnn_probabiltiies.pdf', frameon=False, dpi=110, bbox_inches='tight')
plt.close()