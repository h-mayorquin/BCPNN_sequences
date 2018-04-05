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


variance_protocol_single = False
variance_protocol_multiple = True

markersize = 32
linewidth = 10
figsize = (16, 12)

num = 25
trials = 50
failure_treshold = 0.5

############
# General parameters
############
always_learning = False
k_perfect = True
perfect = False
strict_maximum = True
z_transfer = False
diagonal_zero = False

# First we run the training protocol
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.0
tau_m = 0.020
tau_z_pre_ampa = 0.005
tau_z_post_ampa = 0.005
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
resting_time = 0.0
epochs = 3

###################################################################
# Single protocol
###################################################################
if variance_protocol_single:
    # Recall
    T_recall = 4.0
    T_cue = 0.100
    I_cue = 0.0
    n = 1
    loc = 0.350
    min_scale_percentage = 0
    max_scale_percentage = 25.0

    min_scale = loc * (min_scale_percentage / 100.0)
    max_scale = loc * (max_scale_percentage / 100.0)

    scale_vector = np.linspace(min_scale, max_scale, num=num)
    success_vector_scale = np.zeros((num, trials))
    persistence_time_vector_scale = np.zeros((num, trials))
    training_times = np.zeros((num, trials, n_patterns))

    for index, scale in enumerate(scale_vector):
        for trial_index in range(trials):
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
            training_time = np.random.normal(loc=loc, scale=scale, size=n_patterns)
            training_time[training_time <= tau_m] = tau_m
            protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                                     inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

            # Train
            epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)

            sequences = [patterns_indexes]

            aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
            total_sequence_time, mean, std, success, timings = aux

            success_vector_scale[index, trial_index] = success
            persistence_time_vector_scale[index, trial_index] = mean
            training_times[index, trial_index, :] = training_time

    # Plot
    index = 0
    current_palette = sns.color_palette()

    mean = success_vector_scale.mean(axis=1)
    std = success_vector_scale.std(axis=1)

    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_subplot(111)
    normalized_scale = 100 * scale_vector / loc
    ax.plot(normalized_scale, mean, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
            label='success')
    ax.fill_between(normalized_scale, mean - std, mean + std, color=current_palette[index], alpha=0.25)

    ax.set_xlabel(r'$\frac{\sigma}{\mu} x 100$')
    ax.set_ylabel(r'$\%Success$')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.set_title('Variance protocol with trials = ' + str(trials))

    fig1.savefig('./plot_producers/variance_protocol.pdf', frameon=False, dpi=110, bbox_inches='tight')

    persistence_masked = np.ma.masked_less_equal(persistence_time_vector_scale, 0.0)
    persistence = persistence_masked.mean(axis=1)
    p_std = persistence_masked.std(axis=1)

    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111)
    ax2.plot(normalized_scale, persistence, 'o-', markersize=markersize, linewidth=linewidth)
    ax2.fill_between(normalized_scale, persistence - p_std, persistence + p_std,
                     color=current_palette[index], alpha=0.25)

    ax2.set_xlabel(r'$\frac{\sigma}{\mu}$ x 100')
    ax2.set_ylabel(r'$T_{persistence}$')

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')
    ax2.set_title('Variance protocol with trials = ' + str(trials))

    fig2.savefig('./plot_producers/variance_protocol_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

###################################################################
# Multiple means protocol
###################################################################
if variance_protocol_multiple:
    # Recall
    T_recall = 4.0
    T_cue = 0.100
    I_cue = 0.0
    n = 1
    locations = [0.100, 0.200, 0.300]
    min_scale_percentage = 0
    max_scale_percentage = 25.0

    success_vector_scale_list = []
    persistence_time_vector_scale_list = []
    scale_vector_list = []

    for loc in locations:
        min_scale = loc * (min_scale_percentage / 100.0)
        max_scale = loc * (max_scale_percentage / 100.0)

        scale_vector = np.linspace(min_scale, max_scale, num=num)
        success_vector_scale = np.zeros((num, trials))
        persistence_time_vector_scale = np.zeros((num, trials))
        training_times = np.zeros((num, trials, n_patterns))

        for index, scale in enumerate(scale_vector):
            for trial_index in range(trials):
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
                training_time = np.random.normal(loc=loc, scale=scale, size=n_patterns)
                training_time[training_time <= tau_m] = tau_m
                protocol.simple_protocol(patterns_indexes, training_time=training_time, inter_pulse_interval=inter_pulse_interval,
                                         inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

                # Train
                epoch_history = manager.run_network_protocol(protocol=protocol, verbose=False)
                sequences = [patterns_indexes]

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector_scale[index, trial_index] = success
                persistence_time_vector_scale[index, trial_index] = mean
                training_times[index, trial_index, :] = training_time

        scale_vector_list.append(scale_vector)
        success_vector_scale_list.append(success_vector_scale)
        persistence_time_vector_scale_list.append(persistence_time_vector_scale)

    # Plot
    current_palette = sns.color_palette()

    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111)
    aux = zip(locations, success_vector_scale_list, persistence_time_vector_scale_list, scale_vector_list)

    for index, (loc, success_vector_scale, persistence_time_vector_scale, scale_vector) in enumerate(aux):

        mean = success_vector_scale.mean(axis=1)
        std = success_vector_scale.std(axis=1)
        normalized_scale = 100 * scale_vector / loc
        ax.plot(normalized_scale, mean, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
                label=str(loc))
        ax.fill_between(normalized_scale, mean - std, mean + std, color=current_palette[index], alpha=0.25)

        persistence_masked = np.ma.masked_less_equal(persistence_time_vector_scale, 0.0)
        persistence = persistence_masked.mean(axis=1)
        p_std = persistence_masked.std(axis=1)

         # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        p_std.mask = accepted

        ax2.plot(normalized_scale, persistence, 'o-', color=current_palette[index],
                 markersize=markersize, linewidth=linewidth, label=str(loc))
        ax2.fill_between(normalized_scale, persistence - p_std, persistence + p_std,
                         color=current_palette[index], alpha=0.25)

    ax.set_xlabel(r'$\frac{\sigma}{\mu} x 100$')
    ax.set_ylabel(r'$\%Success$')
    ax.set_xlim([-1, max_scale_percentage + 1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.set_title('Variance protocol with trials = ' + str(trials))

    ax2.set_xlabel(r'$\frac{\sigma}{\mu}$ x 100')
    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlim([-1, max_scale_percentage + 1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')
    ax2.set_title('Variance protocol with trials = ' + str(trials))

    ax.legend()
    ax2.legend()

    fig1.savefig('./plot_producers/variance_protocol.pdf', frameon=False, dpi=110, bbox_inches='tight')
    fig2.savefig('./plot_producers/variance_protocol_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

plt.close()
