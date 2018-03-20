import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from network import Protocol, NetworkManager, BCPNNPerfect
from analysis_functions import calculate_recall_time_quantities
from connectivity_functions import simple_bcpnn_matrix

sns.set(font_scale=2.8)
sns.set_style(style='white')

# Network structure
always_learning = True
k_perfect = True
perfect = True
strict_maximum = True
z_transfer = False

# First we run the training protocol
g_w_ampa = 1.0
g_w = 0.0
g_a = 3.0
g_I = 20.0
tau_a = 0.150
G = 1.0
sigma = 0.0
tau_z_pre_ampa = 0.005
tau_z_post_ampa = 0.005
tau_p = 10.0

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'i_ampa', 'i_nmda', 'beta', 'a']

# Protocol
training_time = 0.100
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
resting_time = 0.0
epochs = 3

# Network parameters
w_self = 1.0
w_next = -1
w_rest = -3.0
w_diff = g_w_ampa * (w_self - w_next)

# Build the network
nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, sigma=sigma, G=G,
                  tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p, g_I=g_I,
                  z_transfer=z_transfer, diagonal_zero=False, strict_maximum=strict_maximum, perfect=perfect,
                  k_perfect=k_perfect, always_learning=always_learning)
nn.g_beta = 0.0

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)
nn.w_ampa = w

# Recall
T_recall = 0.450
T_cue = 0.050
sequences = [[i for i in range(n_patterns)]]
n = 1

aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
total_sequence_time, mean, std, success, timings = aux

i_ampa = manager.history['i_ampa']
a = manager.history['a']
time = np.linspace(0, manager.T_total, a.shape[0])

##########
# Plot
##########
patterns = [0, 1, 2]
# patterns = sequences[0]

w_diff = w_self - w_next
sigma = 0.25 * g_w_ampa * w_diff
linewidth = 10
current_palette = sns.color_palette()

fig = plt.figure(figsize=(16, 12))

ax = fig.add_subplot(111)

for index, pattern in enumerate(patterns):
    current = i_ampa[:, pattern] - g_a * a[:, pattern]
    ax.plot(time, current, color=current_palette[index], linewidth=linewidth, label='current ' + str(index))
    ax.fill_between(time, current - sigma, current + sigma, color=current_palette[index], alpha=0.25)


ax.axhline(w_self, ls='--', linewidth=3,label=r'$g_w w_{self}$', color='black')
ax.axhline(w_next, ls='-.', linewidth=3, label=r'$g_w w_{next}$', color='black')
ax.axhline(w_rest, ls=':', linewidth=3, label=r'$g_w w_{rest}$', color='black')
ax.set_title('Currents evolving in time')

ax.legend(loc=3)
ax.set_ylim([-8, 2])

if True:
    ax.axis('off')

fig.savefig('./plot_producers/noise_diagram.pdf', frameon=False, dpi=110, bbox_inches='tight')


