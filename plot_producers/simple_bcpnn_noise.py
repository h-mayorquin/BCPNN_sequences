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

########
# Graphical parameters
########

markersize = 32
linewidth = 10

num = 25
trials = 50

recall_noise1 = True
recall_noise2 = True
minicolumns_noise = False
######
## General parameters
#######
# Network structure
always_learning = True
k_perfect = True
perfect = True
strict_maximum = True
z_transfer = False

# First we run the training protocol
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
G = 1.0
sigma = 0.1
tau_z_pre_ampa = 0.005
tau_z_post_ampa = 0.005
tau_p = 10.0

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's']

# Protocol
training_time = 0.100
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0
resting_time = 0.0
epochs = 3

if recall_noise1:
    # Network parameters
    w_self = 1.0
    w_next = -1.0
    w_rest = -6
    w_diff = g_w_ampa * (w_self  - w_next)

    w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)

    sigma_vector = np.linspace(0, w_diff, num=num)
    success_vector =  np.zeros((num, trials))
    persistent_time_vector = np.zeros((num, trials))

    for index_sigma, sigma in enumerate(sigma_vector):
        for trial in range(trials):
            # Build the network
            nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, sigma=sigma, G=G,
                              tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                              z_transfer=z_transfer, diagonal_zero=False, strict_maximum=strict_maximum, perfect=perfect,
                              k_perfect=k_perfect, always_learning=always_learning)

            # Build the manager
            manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
            nn.w_ampa = w

            # Recall
            T_recall = 2.2
            T_cue = 0.080
            sequences = [[i for i in range(n_patterns)]]
            n = 1

            aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
            total_sequence_time, mean, std, success, timings = aux

            success_vector[index_sigma, trial] = success
            persistent_time_vector[index_sigma, trial] = mean

    successes = success_vector.mean(axis=1) / 100.0
    sigmas_norm = sigma_vector / w_diff
    std = success_vector.std(axis=1) / 100.0
    persistence =  persistent_time_vector.mean(axis=1)

    fig1 = plt.figure(figsize=(16, 12))
    ax = fig1.add_subplot(111)
    ax.plot(sigmas_norm, successes, 'o-', markersize=markersize, linewidth=linewidth, label='success')
    ax.plot(sigmas_norm, persistence, 'o-', markersize=markersize, linewidth=linewidth, label=r'$T_{persistence}$')

    ax.fill_between(sigmas_norm, successes - std, successes + std, color='blue', alpha=0.1)

    ax.set_ylabel('Success')
    ax.set_xlabel(r'$\frac{\sigma}{g_w w_{diff}}$')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.axvline(1, ls='-', color='red', label=r'$g_w w_{diff}$')

    ax.legend()

    fig1.savefig('./plot_producers/recall_noise1.pdf', frameon=False, dpi=110, bbox_inches='tight')

if recall_noise2:
    # Network parameters
    w_self = 2.0
    w_next = -3.0
    w_rest = -6
    w_diff = g_w_ampa * (w_self - w_next)

    w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)

    sigma_vector = np.linspace(0, w_diff, num=num)
    success_vector =  np.zeros((num, trials))
    persistent_time_vector = np.zeros((num, trials))

    for index_sigma, sigma in enumerate(sigma_vector):
        for trial in range(trials):
            # Build the network
            nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, sigma=sigma, G=G,
                              tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                              z_transfer=z_transfer, diagonal_zero=False, strict_maximum=strict_maximum, perfect=perfect,
                              k_perfect=k_perfect, always_learning=always_learning)

            # Build the manager
            manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
            nn.w_ampa = w

            # Recall
            T_recall = 2.2
            T_cue = 0.080
            sequences = [[i for i in range(n_patterns)]]
            n = 1

            aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
            total_sequence_time, mean, std, success, timings = aux

            success_vector[index_sigma, trial] = success
            persistent_time_vector[index_sigma, trial] = mean

    successes = success_vector.mean(axis=1) / 100.0
    sigmas_norm = sigma_vector / w_diff
    std = success_vector.std(axis=1) / 100.0
    persistence =  persistent_time_vector.mean(axis=1)

    fig2 = plt.figure(figsize=(16, 12))
    ax = fig2.add_subplot(111)
    ax.plot(sigmas_norm, successes, 'o-', markersize=markersize, linewidth=linewidth, label='success')
    ax.plot(sigmas_norm, persistence, 'o-', markersize=markersize, linewidth=linewidth, label=r'$T_{persistence}$')

    ax.fill_between(sigmas_norm, successes - std, successes + std, color='blue', alpha=0.1)

    ax.set_ylabel('Success')
    ax.set_xlabel(r'$\frac{\sigma}{g_w w_{diff}}$')

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.axvline(1, ls='-', color='red', label=r'$g_w w_{diff}$')

    ax.legend()

    fig2.savefig('./plot_producers/recall_noise2.pdf', frameon=False, dpi=110, bbox_inches='tight')


plt.close()