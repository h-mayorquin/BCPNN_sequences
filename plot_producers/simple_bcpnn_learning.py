import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from network import Protocol, NetworkManager, BCPNNPerfect
from analysis_functions import get_weights, calculate_recall_time_quantities

from_pattern = 2
to_pattern = 3

sns.set(font_scale=3.5)
sns.set_style(style='white')


markersize = 24
linewidth = 10

run_training_time = True
run_epochs = True
run_minicolumns = True
run_n_patterns = True

mean = False   # Get the max of the rest of the weights
plot_success = True

num = 30

############
# General parameters
############

always_learning = False
strict_maximum = True
perfect = False
z_transfer = False
k_perfect = True
diagonal_zero = False

g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.0
tau_m = 0.020
tau_z_pre_ampa = 0.025
tau_z_post_ampa = 0.025
tau_p = 100.0

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
epochs = 3

# Recall
T_recall = 3.0
T_cue = 0.100
n = 1

#########
# Training times
#########
if run_training_time:
    training_times_vector = np.linspace(0.050, 1.0, num=num)
    w_self_vector_tt = np.zeros_like(training_times_vector)
    w_next_vector_tt = np.zeros_like(training_times_vector)
    w_rest_vector_tt = np.zeros_like(training_times_vector)
    successes = np.zeros_like(training_times_vector)

    for index, training_time_ in enumerate(training_times_vector):

        # Build the network
        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum, perfect=perfect,
                          k_perfect=k_perfect, always_learning=always_learning)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        protocol = Protocol()
        patterns_indexes = [i for i in range(n_patterns)]
        protocol.simple_protocol(patterns_indexes, training_time=training_time_, inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval, epochs=epochs)

        # Train
        epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=mean)

        w_self_vector_tt[index] = w_self
        w_next_vector_tt[index] = w_next
        w_rest_vector_tt[index] = w_rest

        if plot_success:
            sequences = [patterns_indexes]
            aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
            total_sequence_time, mean, std, success, timings = aux
            successes[index] = success

    fig1 = plt.figure(figsize=(16, 12))
    ax1 = fig1.add_subplot(111)

    ax1.plot(training_times_vector, w_self_vector_tt, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
    ax1.plot(training_times_vector, w_next_vector_tt, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
    ax1.plot(training_times_vector, w_rest_vector_tt, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

    if plot_success:
        ax1.plot(training_times_vector, successes / 100, '--', color='black', lw=linewidth, markersize=markersize)

    ax1.set_xlabel('Training times (s)')
    ax1.set_ylabel('Weight')

    ax1.axhline(0, ls='--', color='black')
    ax1.axvline(0, ls='--', color='black')

    ax1.legend()

    fig1.savefig('./plot_producers/simple_bcpnn_training_time.pdf', frameon=False, dpi=110, bbox_inches='tight')

############
# Epochs
############
if run_epochs:
    epochs_vector = np.arange(1, 31, 1, dtype='int')
    w_self_vector_epochs = np.zeros_like(epochs_vector, dtype='float')
    w_next_vector_epochs = np.zeros_like(epochs_vector, dtype='float')
    w_rest_vector_epochs = np.zeros_like(epochs_vector, dtype='float')
    successes = np.zeros_like(epochs_vector, dtype='float')

    for index, epochs_ in enumerate(epochs_vector):

        # Build the network
        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum, perfect=perfect,
                          k_perfect=k_perfect, always_learning=always_learning)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        protocol = Protocol()
        patterns_indexes = [i for i in range(n_patterns)]
        protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval, epochs=epochs_)

        # Train
        epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=mean)
        w_self_vector_epochs[index] = w_self
        w_next_vector_epochs[index] = w_next
        w_rest_vector_epochs[index] = w_rest

        if plot_success:
            sequences = [patterns_indexes]
            aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
            total_sequence_time, mean, std, success, timings = aux
            successes[index] = success

    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111)
    ax2.plot(epochs_vector, w_self_vector_epochs, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
    ax2.plot(epochs_vector, w_next_vector_epochs, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
    ax2.plot(epochs_vector, w_rest_vector_epochs, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

    if plot_success:
        ax2.plot(epochs_vector, successes / 100, '--', color='black', lw=linewidth, markersize=markersize)

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Weight')

    ax2.axhline(0, ls='--', color='black')
    ax2.axvline(0, ls='--', color='black')

    ax2.legend()
    fig2.savefig('./plot_producers/simple_bcpnn_epochs.pdf', frameon=False, dpi=110, bbox_inches='tight')

#######
# Mincolumns
########
if run_minicolumns:
    minicolumns_vector = np.arange(10, 82, 2, dtype='int')
    w_self_vector_minicolumns = np.zeros_like(minicolumns_vector, dtype='float')
    w_next_vector_minicolumns = np.zeros_like(minicolumns_vector, dtype='float')
    w_rest_vector_minicolumns = np.zeros_like(minicolumns_vector, dtype='float')

    for index, minicolumns_ in enumerate(minicolumns_vector):

        # Build the network
        nn = BCPNNPerfect(hypercolumns, minicolumns_, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum, perfect=perfect,
                          k_perfect=k_perfect, always_learning=always_learning)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        protocol = Protocol()
        patterns_indexes = [i for i in range(minicolumns_)]
        protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval, epochs=epochs,
                                 resting_time=inter_sequence_interval)

        # Train
        epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=mean)
        w_self_vector_minicolumns[index] = w_self
        w_next_vector_minicolumns[index] = w_next
        w_rest_vector_minicolumns[index] = w_rest

    fig3 = plt.figure(figsize=(16, 12))
    ax3 = fig3.add_subplot(111)
    ax3.plot(minicolumns_vector, w_self_vector_minicolumns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
    ax3.plot(minicolumns_vector, w_next_vector_minicolumns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
    ax3.plot(minicolumns_vector, w_rest_vector_minicolumns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

    ax3.set_xlabel('Number of units')
    ax3.set_ylabel('Weight')

    ax3.axhline(0, ls='--', color='black')
    ax3.axvline(0, ls='--', color='black')

    ax3.legend()
    fig3.savefig('./plot_producers/simple_bcpnn_minicolumns.pdf', frameon=False, dpi=110, bbox_inches='tight')

########
# n_patterns
########
if run_n_patterns:

    n_patterns_vector = np.arange(10, 82, 2, dtype='int')
    w_self_vector_patterns = np.zeros_like(n_patterns_vector, dtype='float')
    w_next_vector_patterns = np.zeros_like(n_patterns_vector, dtype='float')
    w_rest_vector_patterns = np.zeros_like(n_patterns_vector, dtype='float')

    for index, n_patterns_ in enumerate(n_patterns_vector):
        # Build the network

        nn = BCPNNPerfect(hypercolumns, minicolumns=80, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum, perfect=perfect,
                          k_perfect=k_perfect, always_learning=always_learning)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        protocol = Protocol()
        patterns_indexes = [i for i in range(n_patterns_)]
        protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval, epochs=epochs,
                                 resting_time=inter_sequence_interval)

        # Train
        epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=mean)
        w_self_vector_patterns[index] = w_self
        w_next_vector_patterns[index] = w_next
        w_rest_vector_patterns[index] = w_rest

    fig4 = plt.figure(figsize=(16, 12))
    ax4 = fig4.add_subplot(111)
    ax4.plot(n_patterns_vector, w_self_vector_patterns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{self}$')
    ax4.plot(n_patterns_vector, w_next_vector_patterns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{next}$')
    ax4.plot(n_patterns_vector, w_rest_vector_patterns, 'o-', lw=linewidth, markersize=markersize, label=r'$w_{rest}$')

    ax4.set_xlabel('Number of patterns')
    ax4.set_ylabel('Weight')

    ax4.axhline(0, ls='--', color='black')
    ax4.axvline(0, ls='--', color='black')

    ax4.legend()
    fig4.savefig('./plot_producers/simple_bcpnn_patterns.pdf', frameon=False, dpi=110, bbox_inches='tight')


plt.close()