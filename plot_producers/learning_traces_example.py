import pprint
import subprocess
import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from network import Protocol, NetworkManager, BCPNNPerfect

run_training_time = False
run_epochs = False
run_minicolumns = True
run_n_patterns = True

sns.set(font_scale=3.5)
sns.set_style(style='white')

# General paramters
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.0
tau_z_pre = 0.050

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'i_ampa', 'i_nmda']

# Protocol
training_time = 0.100
inter_sequence_interval = 0
inter_pulse_interval = 0.0
epochs = 1

# Build the network
nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                  sigma=sigma, G=G, tau_z_pre=tau_z_pre,
                z_transfer=False, diagonal_zero=False, strict_maximum=True, perfect=True)



# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
nn.z_pre = np.zeros(nn.n_units)
# Build the protocol for training
protocol = Protocol()
patterns_indexes = [i for i in range(n_patterns)]
protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True)

######
# Plot
######


o = manager.history['o']
z = manager.history['z_pre']

linewidth = 10
time = np.arange(0, manager.T_total, dt)

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(time, o[:, 3], linewidth=linewidth, ls='--', color='black', label=r'$o_1$')
ax1.plot(time, o[:, 4], linewidth=linewidth, ls='-', color='black', label=r'$o_2$')

y1 = z[:, 3]
y2 = z[:, 4]

ax2.plot(time, y1, linewidth=linewidth, ls='--', color='black', label=r'$z_{1}$')
ax2.plot(time, y2, linewidth=linewidth, ls='-', color='black', label=r'$z_{2}$')

ax2.fill_between(time, y1, 0, where=y1 <= y2 + 0.03, interpolate=True, step='post', color='red', label='co-activation')
ax2.fill_between(time, y2, 0, where=y2 <= y1, interpolate=True, step='post', color='red')

ax2.legend()
ax1.legend()

if True:
    ax1.axis('off')
    ax2.axis('off')

fig.savefig('./plot_producers/traces_example.pdf', frameon=False, dpi=110, bbox_inches='tight')
