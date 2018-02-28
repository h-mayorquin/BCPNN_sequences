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

sns.set(font_scale=2.8)
sns.set_style(style='white')


fig = plt.figure(figsize=(16, 12))
from_pattern = 2
to_pattern = 3

annotations = False
captions = True


# First we run the training protocol
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.0
tau_z_pre_ampa = 0.020
tau_p = 10.0

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre_ampa', 'w_ampa']

# Protocol
training_time = 0.100
inter_sequence_interval = 0.5
inter_pulse_interval = 0.0
epochs = 2


# Build the network
nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_pre_ampa, tau_p=tau_p,
                  z_transfer=False, diagonal_zero=False, strict_maximum=False, perfect=True)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Build the protocol for training
protocol = Protocol()
patterns_indexes = [i for i in range(n_patterns)]
protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
epoch_history = manager.run_network_protocol(protocol=protocol, verbose=True)

z_training = manager.history['z_pre_ampa']
o_training = manager.history['o']
w_training = manager.history['w_ampa'][:, to_pattern, from_pattern]
time_training = np.linspace(0, manager.T_total, num=o_training.shape[0])


##########
# Recall
###########
T_recall = 0.230
T_cue = 0.100
sequences = [patterns_indexes]
I_cue = 0.0
n = 1

nn.strict_maximum = True
aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
total_sequence_time, mean, std, success, timings = aux

o_recall = manager.history['o']
time_recall = np.linspace(0, manager.T_total, num=o_recall.shape[0])

##############
# Plot everything
##############
if captions:
    size = 35
    aux_x = 0.05
    aux_x2 = 0.50
    fig.text(aux_x, 0.93, 'a)', size=size)
    fig.text(aux_x, 0.60, 'c)', size=size)
    fig.text(aux_x, 0.30, 'd)', size=size)
    fig.text(aux_x2, 0.93, 'b)', size=size)
    fig.text(aux_x2, 0.30, 'f)', size=size)
    # fig.text(0.5, 0.40, 'e)', size=size)

gs = gridspec.GridSpec(3, 2)
norm = matplotlib.colors.Normalize(0, n_patterns)
cmap = matplotlib.cm.inferno_r
ax11 = fig.add_subplot(gs[0, 0])
ax12 = fig.add_subplot(gs[0, 1])
ax21 = fig.add_subplot(gs[1, 0])

ax31 = fig.add_subplot(gs[2, 0])

widths = []
for pattern in patterns_indexes:
    width = 10 - pattern * 1.0
    widths.append(width)
    if pattern == 0:
        label = 'Cue'
    else:
        label = str(pattern)

    ax12.plot(time_training, o_training[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax31.plot(time_recall, o_recall[:, pattern], color=cmap(norm(pattern)), linewidth=width)

# The learning part
ax11.plot(time_training, z_training[:, from_pattern], color=cmap(norm(from_pattern)), linewidth=widths[from_pattern],
          label=r'$z_{2}$')
ax11.plot(time_training, z_training[:, to_pattern], color=cmap(norm(to_pattern)), linewidth=widths[to_pattern],
          label=r'$z_{3}$')

z_product = z_training[:, from_pattern] * z_training[:, to_pattern]
z_mask = z_product > 0.10

ax21.plot(time_training, w_training, linewidth=10, label=r'$w_{next}$')
ax21.axhline(0, ls='--', color='black')

alpha = 0.5
ax21.fill_between(time_training, np.min(w_training)*1.1, np.max(w_training)*1.1,
                  where=z_mask, facecolor='red', alpha=alpha, label='co-activation')
ax11.fill_between(time_training, np.min(z_training[:, to_pattern]) *1.1, np.max(z_training[:, to_pattern])*1.1,
                  where=z_mask, facecolor='red', alpha=alpha)


# Labels, and legends
ax11.legend()
ax11.set_xlabel('Time (ms)')
ax21.set_xlabel('Time (ms)')
ax12.set_xlabel('Time (ms)')
ax31.set_xlabel('Time (ms)')

ax11.set_title('z-traces')
ax31.set_title('Recall')
ax21.set_title('Weight evolution')
ax12.set_title('Training')


fig.tight_layout()
##############
# The weight matrix and recall
##############
cmap = matplotlib.cm.inferno

rect = [0.55, 0.10, 0.33, 0.33]

ax_conn = fig.add_axes(rect)
#ax_conn = fig.add_subplot(gs[2, 1])

im = ax_conn.imshow(nn.w_ampa, cmap=cmap)

ax_conn.set_xlabel('pre-synaptic')
ax_conn.set_ylabel('post-synaptic')
ax_conn.xaxis.set_ticklabels([])
ax_conn.yaxis.set_ticklabels([])

divider = make_axes_locatable(ax_conn)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax_conn.set_title('Connectivity Matrix')

if annotations:
    ax_conn.annotate(r'$w_{next}$', xy=(0, 1), xytext=(0, 5),
                     arrowprops=dict(facecolor='red', shrink=0.15))

    ax_conn.annotate(r'$w_{self}$', xy=(0.0, 0), xytext=(4, -1),
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{rest}$', xy=(7, 2.5), xytext=(7, 5),
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{rest}$', xy=(2, 8), xytext=(2, 6.5),
                arrowprops=dict(facecolor='red', shrink=0.05))

######
#
######

handles, labels = ax21.get_legend_handles_labels()

fig.legend(handles=handles, labels=labels, loc=(0.39, 0.46), fancybox=False, frameon=True, facecolor=(1.0, 1.0, 1.0),
           fontsize=22, ncol=1)

handles, labels = ax12.get_legend_handles_labels()

fig.legend(handles=handles, labels=labels, loc=(0.64, 0.46), fancybox=True, frameon=True, facecolor=(0.9, 0.9, 0.9),
           fontsize=22, ncol=2)


fig.savefig('./plot_producers/bcpnn_learning_and_recall.pdf', frameon=False, dpi=110, bbox_inches='tight')
plt.close()
