import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import sys
sys.path.append('../')

np.set_printoptions(suppress=True, precision=2)

sns.set(font_scale=3.0)
sns.set_style(style='white')

from network import Protocol, BCPNNModular, NetworkManager, BCPNNPerfect
from plotting_functions import plot_weight_matrix, plot_state_variables_vs_time, plot_winning_pattern
from plotting_functions import plot_network_activity, plot_network_activity_angle
from analysis_functions import calculate_recall_time_quantities, calculate_angle_from_history
from connectivity_functions import artificial_connectivity_matrix


def simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest):

    w = np.ones((minicolumns, minicolumns)) * w_rest
    for i in range(minicolumns):
        w[i, i] = w_self

    for i in range(minicolumns -1):
        w[i + 1, i] = w_next

    return w


g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
tau_z = 0.150
G = 1.0
sigma = 0.0

perfect = True
strict_maximum = True

w_self = 1.0
w_next = -2
w_rest = -4

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'a', 'i_ampa']

nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                  sigma=sigma, G=G,
                  z_transfer=False, diagonal_zero=False, strict_maximum=strict_maximum, perfect=perfect)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)
nn.w_ampa = w

# Recall
T_recall = 2.2
T_cue = 0.080
sequences = [[i for i in range(n_patterns)]]
n = 1

aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
total_sequence_time, mean, std, success, timings = aux

# Extract quantities
norm = matplotlib.colors.Normalize(0, n_patterns)
cmap = matplotlib.cm.inferno_r

o = manager.history['o']
a = manager.history['a']
i_ampa = manager.history['i_ampa']

T_total = manager.T_total
time = np.arange(0, T_total, dt)

# Plot
captions = True
annotations = True

gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(22, 12))

# Captions
if captions:
    size = 35
    aux_x = 0.04
    fig.text(aux_x, 0.95, 'a)', size=size)
    fig.text(aux_x, 0.60, 'b)', size=size)
    fig.text(aux_x, 0.35, 'c)', size=size)
    fig.text(0.55, 0.90, 'd)', size=size)
    # fig.text(0.5, 0.40, 'e)', size=size)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

for pattern in range(n_patterns):
    width = 10.0 - pattern * 1.0
    if pattern == 0:
        label = 'Cue'
    else:
        label = str(pattern)

    ax1.plot(time, o[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax2.plot(time, a[:, pattern], color=cmap(norm(pattern)), linewidth=width, label=label)
    ax3.plot(time, i_ampa[:, pattern] - g_a * a[:, pattern] , color=cmap(norm(pattern)),
            linewidth=width)


ax1.set_title('Unit activity')
ax2.set_title('Adaptation current')
ax3.set_title('Self-Exc current minus adaptation')

ax3.axhline(g_w_ampa * w_next, ls='--', color='black', label=r'$w_{next} o$')
ax3.legend()
ax3.set_xlabel('Time (s)')
fig.tight_layout()

# Here we plot our connectivity matrix
rect = [0.46, 0.48, 0.40, 0.40]
# ax_conn = fig.add_subplot(gs[:2, 1])
ax_conn = fig.add_axes(rect)

cmap = matplotlib.cm.inferno_r
im = ax_conn.imshow(w, cmap=cmap)

ax_conn.set_xlabel('pre-synaptic')
ax_conn.set_ylabel('post-synaptic')
ax_conn.xaxis.set_ticklabels([])
ax_conn.yaxis.set_ticklabels([])

divider = make_axes_locatable(ax_conn)
cax = divider.append_axes('right', size='5%', pad=0.05)

if annotations:
    ax_conn.annotate(r'$w_{next}$', xy=(0, 1), xytext=(0, 5),
                     arrowprops=dict(facecolor='red', shrink=0.15))

    ax_conn.annotate(r'$w_{self}$', xy=(0.0, 0), xytext=(4, -1),
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{rest}$', xy=(7, 2.5), xytext=(7, 5),
                arrowprops=dict(facecolor='red', shrink=0.05))

    ax_conn.annotate(r'$w_{rest}$', xy=(2, 8), xytext=(2, 6.5),
                arrowprops=dict(facecolor='red', shrink=0.05))


fig.colorbar(im, cax=cax, orientation='vertical')

# Let's plot our legends
# ax_legend = fig.add_subplot(gs[2, 1])
# lines = ax1.get_lines()
handles, labels = ax1.get_legend_handles_labels()
# ax_legend.legend(ax1.get_legend_handles_labels())

fig.legend(handles=handles, labels=labels, loc=(0.65, 0.09), fancybox=True, frameon=True, facecolor=(0.9, 0.9, 0.9),
           fontsize=28, ncol=2)


# plt.show()
fig.savefig('./plot_producers/simple_bcpnn_recall.pdf', frameon=False, dpi=110, bbox_inches='tight')
plt.close()