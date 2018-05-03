import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from network import Protocol, NetworkManager, BCPNNPerfect, TimedInput
from connectivity_functions import create_orthogonal_canonical_representation, build_network_representation
from connectivity_functions import get_weights_from_probabilities, get_probabilities_from_network_representation
from analysis_functions import calculate_recall_time_quantities, get_weights
from analysis_functions import get_weights_collections


def generate_plot_for_variable(filename, x_values, xlabel):
    format = '.pdf'
    folder = './plot_producers/off_line_rule_learning_'

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x_values, w_self_vector, 'o-', markersize=markersize, linewidth=linewidth, label=r'$w_{self}$')
    ax.plot(x_values, w_next_vector, 'o-', markersize=markersize, linewidth=linewidth, label=r'$w_{next}$')
    ax.plot(x_values, w_rest_vector, 'o-', markersize=markersize, linewidth=linewidth, label=r'$w_{rest}$')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_ylabel(r'$w$')
    ax.set_xlabel(xlabel)
    ax.legend()

    type = 'w'
    aux_filename = folder + filename + type + format
    fig.savefig(aux_filename, frameon=False, dpi=110, bbox_inches='tight')
    fig.clear()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.plot(x_values, factor * Pij_self_vector, 'o-', markersize=markersize, linewidth=linewidth,
            label=r'$P_{self}$')
    ax.plot(x_values, factor * Pij_next_vector, 'o-', markersize=markersize, linewidth=linewidth,
            label=r'$P_{next}$')
    ax.plot(x_values, factor * Pij_rest_vector, 'o-', markersize=markersize, linewidth=linewidth,
            label=r'$P_{rest}$')
    ax.plot(x_values, factor * pi_self_vector, 'o-', markersize=markersize, linewidth=linewidth,
            label=r'$p_i * p_j s$', color='black')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_ylabel(r'Probabilities')
    ax.set_xlabel(xlabel)
    ax.legend()

    type = 'p'
    aux_filename = folder + filename + type + format
    fig.savefig(aux_filename, frameon=False, dpi=110, bbox_inches='tight')
    fig.clear()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(x_values, persistence_time_vector, 'o-', markersize=markersize,
            linewidth=linewidth, label=r'$T_{persistence}$')
    ax.plot(x_values, success_vector / 100.0, 'o-', markersize=markersize,
            linewidth=linewidth, label=r'Success')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_ylabel(r'$T_{persistence} (s)$')
    ax.set_xlabel(xlabel)
    ax.legend()

    type = 'time'
    aux_filename = folder + filename + type + format
    fig.savefig(aux_filename, frameon=False, dpi=110, bbox_inches='tight')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.plot(x_values[:-1], np.diff(Pij_self_vector) / np.abs(Pij_self_vector[:-1]),
            'o-', markersize=markersize, linewidth=linewidth, label=r'$P_{self}$', alpha=alpha)
    ax.plot(x_values[:-1], np.diff(Pij_next_vector) / np.abs(Pij_next_vector[:-1]), 'o-',
            markersize=markersize, linewidth=linewidth, label=r'$P_{next}$', alpha=alpha)
    ax.plot(x_values[:-1], np.diff(Pij_rest_vector) / np.abs(Pij_rest_vector[:-1]),
            'o-', markersize=markersize, linewidth=linewidth, label=r'$P_{rest}$', alpha=alpha)
    ax.plot(x_values[:-1], np.diff(pi_self_vector) / np.abs(pi_self_vector[:-1]), 'o-', alpha=alpha,
            markersize=markersize, linewidth=linewidth, color='black', label=r'$p_i * p_j$')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_ylabel(r'$\Delta $ Probabilities')
    ax.set_xlabel(xlabel)
    ax.legend()

    type = 'diff'
    aux_filename = folder + filename + type + format
    fig.savefig(aux_filename, frameon=False, dpi=110, bbox_inches='tight')

    plt.close()

sns.set(font_scale=2.8)
sns.set_style(style='white')

epsilon = 10e-10
from_pattern = 2
to_pattern = 3

figsize = (16, 12)
markersize = 25
linewidth = 10
factor = 1.0
alpha = 0.8
normal_palette = sns.color_palette()

plot_training_time = False
plot_tau_z = False
plot_resting_time = False
plot_epochs = False
plot_inter_sequence_time = False
plot_inter_pulse_interval = False
plot_minicolumns_fixed = True
plot_minicolumns_var = True
plot_hypercolumns = False

#####################
# General parameters
#####################

always_learning = False
strict_maximum = True
perfect = False
z_transfer = False
k_perfect = True
diagonal_zero = False
normalized_currents = True

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
resting_time = 3.0

# Recall
T_recall = 3.0
n = 1
T_cue = 0.050

##############################
# Training time
##############################
if plot_training_time:
    epsilon_ = epsilon
    num = 20
    training_times = np.linspace(0.050, 1.0, num=num)
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)

    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, training_time_ in enumerate(training_times):

        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time_,
                                inter_pulse_interval=inter_pulse_interval,
                                inter_sequence_interval=inter_sequence_interval,
                                epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon_)

        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'training_time_'
    x_values = training_times
    xlabel = r'Training Time (s)'
    generate_plot_for_variable(filename, x_values, xlabel)

##############################
# tau_z
###############################
if plot_tau_z:
    num = 15
    tau_z_vector = np.linspace(0.025, 0.250, num=num)
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, tau_z_pre_ampa_ in enumerate(tau_z_vector):

        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                inter_pulse_interval=inter_pulse_interval,
                                inter_sequence_interval=inter_sequence_interval,
                                epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa_)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa_)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon)

        # Patterns parameters
        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa_, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'tau_z_'
    x_values = tau_z_vector
    xlabel = r'$\tau_z$ (s)'
    generate_plot_for_variable(filename, x_values, xlabel)

#########################
# Resting time
#########################
if plot_resting_time:
    num = 15
    resting_times = np.linspace(0.0, 3.0, num=num)
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, resting_time_ in enumerate(resting_times):

        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                inter_pulse_interval=inter_pulse_interval,
                                inter_sequence_interval=inter_sequence_interval,
                                epochs=epochs, resting_time=resting_time_)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon)


        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'resting_'
    x_values = resting_times
    xlabel = r'Resting time (s)'
    generate_plot_for_variable(filename, x_values, xlabel)

###########################
# Epochs
###########################

if plot_epochs:
    num = 15
    epochs_vector = np.arange(1, 10, 1, dtype='int')
    success_vector = np.zeros(epochs_vector.size)
    persistence_time_vector = np.zeros(epochs_vector.size)
    w_self_vector = np.zeros(epochs_vector.size)
    w_next_vector = np.zeros(epochs_vector.size)
    w_rest_vector = np.zeros(epochs_vector.size)

    pi_self_vector = np.zeros(epochs_vector.size)
    Pij_self_vector = np.zeros(epochs_vector.size)
    pi_next_vector = np.zeros(epochs_vector.size)
    Pij_next_vector = np.zeros(epochs_vector.size)
    pi_rest_vector = np.zeros(epochs_vector.size)
    Pij_rest_vector = np.zeros(epochs_vector.size)

    for index, epochs_ in enumerate(epochs_vector):
        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                 inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval,
                                 epochs=epochs_, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon)

        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'epochs_'
    x_values = epochs_vector
    xlabel = r'Epochs'
    generate_plot_for_variable(filename, x_values, xlabel)


#############################
# Inter-sequence times
#############################
if plot_inter_sequence_time:
    num = 15
    inter_sequence_times = np.linspace(0.0, 3.0, num=num)
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, inter_sequence_interval_ in enumerate(inter_sequence_times):

        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                inter_pulse_interval=inter_pulse_interval,
                                inter_sequence_interval=inter_sequence_interval_,
                                epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon)

        # Patterns parameters
        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'ISI_'
    x_values = inter_sequence_times
    xlabel = r'$ISI (s)$'
    generate_plot_for_variable(filename, x_values, xlabel)

##########################
# Inter Pulse Interval
##########################
if plot_inter_pulse_interval:
    num = 15
    inter_pulse_times = np.linspace(0.0, 1.0, num=num)
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, inter_pulse_interval_ in enumerate(inter_pulse_times):
        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                 inter_pulse_interval=inter_pulse_interval_,
                                 inter_sequence_interval=inter_sequence_interval,
                                 epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns, epsilon)

        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'IPI_'
    x_values = inter_pulse_times
    xlabel = r'$IPI (s)$'
    generate_plot_for_variable(filename, x_values, xlabel)

########################
# Minicolumns
########################

if plot_minicolumns_fixed:
    num = 20
    minicolumns_vector = np.linspace(10, 100, num=num, dtype='int')
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, minicolumns_ in enumerate(minicolumns_vector):

        matrix = create_orthogonal_canonical_representation(minicolumns_, hypercolumns)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns_, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                inter_pulse_interval=inter_pulse_interval,
                                inter_sequence_interval=inter_sequence_interval,
                                epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns_, hypercolumns, epsilon)


        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns, minicolumns_, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        T_recall = 0.200 * n_patterns
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)
        w_rest  = w_timed[to_pattern + 1, from_pattern]

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'minicolumns_fixed_'
    x_values = minicolumns_vector
    xlabel = r'Minicolumns'
    generate_plot_for_variable(filename, x_values, xlabel)

###############################
# Minicolumns Variable
###############################
if plot_minicolumns_var:
    num = 20
    minicolumns_vector = np.linspace(10, 100, num=num, dtype='int')
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, minicolumns_ in enumerate(minicolumns_vector):
        n_patterns_ = minicolumns_
        matrix = create_orthogonal_canonical_representation(minicolumns_, hypercolumns)[:n_patterns_]
        network_representation = build_network_representation(matrix, minicolumns_, hypercolumns)

        timed_input = TimedInput(network_representation, dt, training_time,
                                 inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval,
                                 epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns_, hypercolumns, epsilon)

        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns, minicolumns_, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns_)]
        sequences = [patterns_indexes]
        T_recall = 0.200 * n_patterns_
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'minicolumns_var_'
    x_values = minicolumns_vector
    xlabel = r'Minicolumns'
    generate_plot_for_variable(filename, x_values, xlabel)

#######################
# Hypercolumns
#######################
if plot_hypercolumns:
    num = 10
    hypercolumns_vector = np.linspace(1, 10, num=num, dtype='int')
    success_vector = np.zeros(num)
    persistence_time_vector = np.zeros(num)
    w_self_vector = np.zeros(num)
    w_next_vector = np.zeros(num)
    w_rest_vector = np.zeros(num)

    pi_self_vector = np.zeros(num)
    Pij_self_vector = np.zeros(num)
    pi_next_vector = np.zeros(num)
    Pij_next_vector = np.zeros(num)
    pi_rest_vector = np.zeros(num)
    Pij_rest_vector = np.zeros(num)

    for index, hypercolumns_ in enumerate(hypercolumns_vector):
        matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns_)[:n_patterns]
        network_representation = build_network_representation(matrix, minicolumns, hypercolumns_)

        timed_input = TimedInput(network_representation, dt, training_time,
                                 inter_pulse_interval=inter_pulse_interval,
                                 inter_sequence_interval=inter_sequence_interval,
                                 epochs=epochs, resting_time=resting_time)

        S = timed_input.build_timed_input()
        z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
        z_post = timed_input.build_filtered_input_post(tau_z_pre_ampa)

        pi, pj, P = timed_input.calculate_probabilities_from_time_signal(filtered=True)
        w_timed, beta_timed = get_weights_from_probabilities(pi, pj, P, minicolumns, hypercolumns_, epsilon)

        # Patterns parameters

        nn = BCPNNPerfect(hypercolumns_, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                          sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                          z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                          perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                          normalized_currents=normalized_currents)

        # Build the manager
        manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

        # Build the protocol for training
        nn.w_ampa = w_timed

        # Recall
        patterns_indexes = [i for i in range(n_patterns)]
        sequences = [patterns_indexes]
        # manager.run_network_recall(T_recall=1.0, T_cue=0.100, I_cue=0, reset=True, empty_history=True)
        aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
        total_sequence_time, mean, std, success, timings = aux
        w_self, w_next, w_rest = get_weights(manager, from_pattern, to_pattern, mean=False)
        w_rest = w_timed[to_pattern + 1, from_pattern]

        success_vector[index] = success
        persistence_time_vector[index] = mean
        w_self_vector[index] = w_self
        w_next_vector[index] = w_next
        w_rest_vector[index] = w_rest

        pi_self_vector[index] = pi[from_pattern] * pj[from_pattern]
        Pij_self_vector[index] = P[from_pattern, from_pattern]
        pi_next_vector[index] = pi[from_pattern] * pj[to_pattern]
        Pij_next_vector[index] = P[to_pattern, from_pattern]
        pi_rest_vector[index] = pi[from_pattern] * pj[to_pattern + 1]
        Pij_rest_vector[index] = P[to_pattern + 1, from_pattern]

    # Plot
    filename = 'hypercolumns_'
    x_values = hypercolumns_vector
    xlabel = r'Hypercolumns'
    generate_plot_for_variable(filename, x_values, xlabel)

plt.close()