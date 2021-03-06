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
from analysis_functions import get_weights

sns.set(font_scale=2.8)
sns.set_style(style='white')

markersize = 25
linewidth = 10

from_pattern = 1
to_pattern = 2

############
# General parameters
############
always_learning = False
k_perfect = True
perfect = False
strict_maximum = True
z_transfer = False
diagonal_zero = False

plot_success = False

num = 30

# First we run the training protocol
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.0
tau_m = 0.020
tau_z_pre_ampa = 0.025
tau_z_post_ampa = 0.025
tau_p = 10.0

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o']

# Protocol
training_time = 0.100
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
resting_time = inter_sequence_interval
epochs = 3

# Recall
T_cue = 0.100
T_recall = 3.0
n = 1

tau_z_vector = np.linspace(0.005, 0.250, num=num)
w_self_vector_tau_z = np.zeros_like(tau_z_vector)
w_next_vector_tau_z = np.zeros_like(tau_z_vector)
w_rest_vector_tau_z = np.zeros_like(tau_z_vector)

successes = np.zeros_like(tau_z_vector)

for index, tau_z_pre_ampa_ in enumerate(tau_z_vector):

    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                      sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa_, tau_z_post_ampa=tau_z_pre_ampa_, tau_p=tau_p,
                      z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum, perfect=perfect,
                      k_perfect=k_perfect, always_learning=always_learning)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build the protocol for training
    protocol = Protocol()
    patterns_indexes = [i for i in range(n_patterns)]
    protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

    # Train
    epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

    w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern)

    w_self_vector_tau_z[index] = w_self
    w_next_vector_tau_z[index] = w_next
    w_rest_vector_tau_z[index] = w_rest

    if plot_success:
        sequences = [patterns_indexes]
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        successes[index] = success


fig1 = plt.figure(figsize=(16, 12))
ax1 = fig1.add_subplot(111)
ax1.plot(tau_z_vector, w_self_vector_tau_z, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
ax1.plot(tau_z_vector, w_next_vector_tau_z, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
ax1.plot(tau_z_vector, w_rest_vector_tau_z, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

if plot_success:
    ax1.plot(tau_z_vector, successes / 100, '--', color='black', lw=linewidth, markersize=markersize)

ax1.set_xlabel(r'$\tau_z$(s)')
ax1.set_ylabel('Weight')

ax1.axhline(0, ls='--', color='black')
ax1.axvline(0, ls='--', color='black')

ax1.legend()
fig1.savefig('./plot_producers/bcpnn_learning_tau_z.pdf', frameon=False, dpi=110, bbox_inches='tight')

tau_p_vector = np.linspace(1, 30, num=num)
w_self_vector_tau_p = np.zeros_like(tau_p_vector)
w_next_vector_tau_p = np.zeros_like(tau_p_vector)
w_rest_vector_tau_p = np.zeros_like(tau_p_vector)
successes = np.zeros_like(tau_p_vector)

for index, tau_p_ in enumerate(tau_p_vector):

    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                      sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p_,
                      z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                      perfect=perfect, k_perfect=k_perfect, always_learning=always_learning)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Build the protocol for training
    protocol = Protocol()
    patterns_indexes = [i for i in range(n_patterns)]
    protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

    # Train
    epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

    w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern)

    w_self_vector_tau_p[index] = w_self
    w_next_vector_tau_p[index] = w_next
    w_rest_vector_tau_p[index] = w_rest

    if plot_success:
        sequences = [patterns_indexes]
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        successes[index] = success

fig2 = plt.figure(figsize=(16, 12))
ax2 = fig2.add_subplot(111)
ax2.plot(tau_p_vector, w_self_vector_tau_p, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
ax2.plot(tau_p_vector, w_next_vector_tau_p, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
ax2.plot(tau_p_vector, w_rest_vector_tau_p, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

if plot_success:
    ax2.plot(tau_p_vector, successes / 100, '--', color='black', lw=linewidth, markersize=markersize)

ax2.set_xlabel(r'$\tau_p$ (s)')
ax2.set_ylabel('Weight')

ax2.axhline(0, ls='--', color='black')
ax2.axvline(0, ls='--', color='black')

ax2.legend()
fig2.savefig('./plot_producers/bcpnn_learning_tau_p.pdf', frameon=False, dpi=110, bbox_inches='tight')

plt.close()

