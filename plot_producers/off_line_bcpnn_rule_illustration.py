import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from connectivity_functions import create_orthogonal_canonical_representation, build_network_representation
from connectivity_functions import get_weights_from_probabilities, get_probabilities_from_network_representation

from network import TimedInput

sns.set(font_scale=2.8)
sns.set_style(style='white')

markersize = 25
linewidth = 5
figsize = (16, 12)

minicolumns = 5
hypercolumns = 2

dt = 0.001
training_time = 0.100
inter_sequence_interval = 0.0
epochs = 1


matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)
network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

timed_input = TimedInput(minicolumns, hypercolumns, network_representation, dt, training_time, inter_pulse_interval=0.0,
                         inter_sequence_interval=inter_sequence_interval, epochs=epochs)
S = timed_input.build_timed_input()
time = timed_input.time
n_units = timed_input.n_units

pi, pij = get_probabilities_from_network_representation(network_representation)
w_static = get_weights_from_probabilities(pi, pi, pij, minicolumns, hypercolumns)

p_pre, p_post, P = timed_input.calculate_probabilities_from_time_signal(filtered=False)
w_timed = get_weights_from_probabilities(p_pre, p_post, P, minicolumns, hypercolumns)

tau_z_pre = 0.050
z_pre = timed_input.build_filtered_input_pre(tau_z_pre)
tau_z_post = 0.001
z_post = timed_input.build_filtered_input_post(tau_z_post)

pi_filtered, pj_filter, P_filtered = timed_input.calculate_probabilities_from_time_signal(filtered=True)
w_filtered = get_weights_from_probabilities(pi_filtered, pj_filter, P_filtered, minicolumns, hypercolumns)

###################
## Plot
###################
cmap = matplotlib.cm.binary
extent = [time[0], time[-1], 0, n_units]

fig1 = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(n_units, 2)

ax1 = fig1.add_subplot(gs[:, 0])
im = ax1.imshow(S, aspect='auto', origin='lower', cmap=cmap, extent=extent)
for index in range(n_units):
    ax1.axhline(index, ls='--', color='gray')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pattern')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

for index, s in enumerate(S):
    ax_n = fig1.add_subplot(gs[n_units - index - 1, 1])
    ax_n.plot(time, s, linewidth=linewidth, markersize=markersize, color='black')
    ax_n.axhline(0, ls='--', color='gray')
    #ax_n.set_axis_off()
    if True:
        ax_n.spines['top'].set_visible(False)
        ax_n.spines['right'].set_visible(False)
        ax_n.spines['bottom'].set_visible(False)
        ax_n.spines['left'].set_visible(False)
        ax_n.get_yaxis().set_ticks([])
    if index==0:
        ax_n.set_xlabel('Time (s)')
    else:
        ax_n.get_xaxis().set_ticks([])

fig1.savefig('./plot_producers/off_line_rule_illustration_signal.pdf', frameon=False, dpi=110, bbox_inches='tight')

################
# The connectivity matrices
###################

fig2 = plt.figure(figsize=figsize)

aspect = 'auto'
cmap = matplotlib.cm.RdBu_r

aux_max = np.max((w_static, w_timed, pij, P))
aux_min = np.max((w_static, w_timed, pij, P))
aux = np.max(np.abs((aux_min, aux_max)))
vmax = aux
vmin = -aux

ax_static_w = fig2.add_subplot(223)
im = ax_static_w.imshow(w_static, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
ax_static_w.set_title(r'$w_{static}$')

ax_time_w = fig2.add_subplot(224)
im = ax_time_w.imshow(w_timed, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
ax_time_w.set_title(r'$w_{time}$')

ax_static_p = fig2.add_subplot(221)
im = ax_static_p.imshow(pij, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
ax_static_p.set_title('Co-activations static')

ax_time_p = fig2.add_subplot(222)
im = ax_time_p.imshow(P, aspect=aspect, cmap=cmap, vmax=vmax, vmin=vmin)
ax_time_p.set_title('Co-activations time')

#fig2.tight_layout()

axes = np.array(fig2.get_axes())
fig2.colorbar(im, ax=axes.ravel().tolist())


fig2.savefig('./plot_producers/off_line_rule_illustration_weight.pdf', frameon=False, dpi=110, bbox_inches='tight')

#######
# Filtered signals
#######
cmap = matplotlib.cm.binary
extent = [time[0], time[-1], 0, n_units]

fig3 = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(n_units, 2)

ax1 = fig3.add_subplot(gs[:, 0])
im = ax1.imshow(z_pre, aspect='auto', origin='lower', cmap=cmap, extent=extent)
for index in range(n_units):
    ax1.axhline(index, ls='--', color='gray')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pattern')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

for index, s in enumerate(z_pre):
    ax_n = fig3.add_subplot(gs[n_units - index - 1, 1])
    ax_n.plot(time, s, linewidth=linewidth, markersize=markersize, color='black')
    ax_n.axhline(0, ls='--', color='gray')
    #ax_n.set_axis_off()
    if True:
        ax_n.spines['top'].set_visible(False)
        ax_n.spines['right'].set_visible(False)
        ax_n.spines['bottom'].set_visible(False)
        ax_n.spines['left'].set_visible(False)
        ax_n.get_yaxis().set_ticks([])
    if index==0:
        ax_n.set_xlabel('Time (s)')
    else:
        ax_n.get_xaxis().set_ticks([])

fig3.savefig('./plot_producers/off_line_rule_illustration_filter.pdf', frameon=False, dpi=110, bbox_inches='tight')

#######
# Filtered weights
########
figsize = (20, 10)
fig4 = plt.figure(figsize=figsize)

aspect = 'auto'
cmap = matplotlib.cm.RdBu_r

aux_max = np.max((w_filtered, P_filtered))
aux_min = np.max((w_filtered, P_filtered))
aux = np.max(np.abs((aux_min, aux_max)))
vmax = aux
vmin = -aux

ax1= fig4.add_subplot(121)
im = ax1.imshow(w_filtered, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
ax_static_w.set_title(r'$w_{filtered}$')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig4.colorbar(im, cax=cax, orientation='vertical')

aux_max = np.max((P_filtered))
aux_min = np.max((P_filtered))
aux = np.max(np.abs((aux_min, aux_max)))
vmax = aux
vmin = -aux

ax2 = fig4.add_subplot(122)
im = ax2.imshow(P_filtered, aspect=aspect, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_title(r'Co-activations')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig4.colorbar(im, cax=cax, orientation='vertical')

fig4.tight_layout()

# axes = np.array(fig4.get_axes())

# fig4.colorbar(im, ax=axes.ravel().tolist())

fig4.savefig('./plot_producers/off_line_rule_illustration_weight_filtered.pdf', frameon=False, dpi=110, bbox_inches='tight')

plt.close()