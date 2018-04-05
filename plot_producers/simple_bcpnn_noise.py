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
figsize = (16, 12)
alpha = 0.25

num = 25
trials = 200
failure_treshold = 0.5  # Only take into account the examples that succed in 70 per cent of the recalls

recall_noise_perfect = True
recall_noise = True
matrix_noise = True
minicolumns_recall_noise = True
minicolumns_matrix_noise = True

######
# General parameters
######

# Network structurepp
always_learning = False
k_perfect = True
strict_maximum = True
z_transfer = False
perfect = False
diagonal_zero = False

# First we run the training protocol
g_w_ampa = 1.5
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

if recall_noise_perfect:
    perfect_ = True
    # Network parameters
    w_diff_set = [1, 2, 3]
    successes_list = []
    persistent_time_list = []
    w_self = 1.0
    w_rest = -10.0

    for w_diff in w_diff_set:
        max_noise = (g_w_ampa * w_diff) / 2.0

        w_next = w_self - w_diff

        w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)

        sigma_vector = np.linspace(0, max_noise, num=num)
        success_vector = np.zeros((num, trials))
        persistent_time_vector = np.zeros((num, trials))

        for index_sigma, sigma in enumerate(sigma_vector):
            for trial in range(trials):
                # Build the network
                nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                                  tau_m=tau_m,
                                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa,
                                  tau_p=tau_p,
                                  z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                                  perfect=perfect_,
                                  k_perfect=k_perfect, always_learning=always_learning)

                # Build the manager
                manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
                nn.w_ampa = w

                # Recall
                T_recall = 3.0
                T_cue = 0.100
                sequences = [[i for i in range(n_patterns)]]
                n = 1

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector[index_sigma, trial] = success
                persistent_time_vector[index_sigma, trial] = mean

        successes_list.append(success_vector)
        persistent_time_list.append(persistent_time_vector)

    # Plot
    fig1 = plt.figure(figsize=figsize)
    ax = fig1.add_subplot(111)

    current_palette = sns.color_palette()

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):

        max_noise = (g_w_ampa * w_diff) / 2.0
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = (2 * sigma_vector) / (g_w_ampa * w_diff)

        successes = success_vector.mean(axis=1)
        std = success_vector.std(axis=1)

        ax.plot(sigmas_norm, successes, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
                label=r'$w_{diff} = $' + str(w_diff))
        ax.fill_between(sigmas_norm, successes - std, successes + std, color=current_palette[index], alpha=alpha)

    ax.set_xlabel(r'$\frac{2 \sigma}{g_w w_{diff}}$')
    ax.set_ylabel('% Success')
    ax.set_xlim([-0.1, 1.1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')
    ax.axvline(1, ls='--', color='red', label=r'$ \frac{g_w w_{diff}}{2}$')

    ax.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax.legend()

    fig1.savefig('./plot_producers/recall_noise_perfect.pdf', frameon=False, dpi=110, bbox_inches='tight')

    fig12 = plt.figure(figsize=figsize)
    ax2 = fig12.add_subplot(111)

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):

        max_noise = (g_w_ampa * w_diff) / 2.0
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = (2 * sigma_vector) / (g_w_ampa * w_diff)

        persistence_masked = np.ma.masked_less_equal(persistent_time_vector, 0.0)
        persistence = persistence_masked.mean(axis=1)
        std_pt = persistence_masked.std(axis=1)

        # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        std_pt.mask = accepted


        ax2.plot(sigmas_norm, persistence, 'o-', color=current_palette[index], markersize=markersize,
                 linewidth=linewidth, label=r'$w_{diff} = $' + str(w_diff))
        ax2.fill_between(sigmas_norm, persistence - std_pt, persistence + std_pt,
                         color=current_palette[index], alpha=alpha)

    ax2.set_xlabel(r'$\frac{2 \sigma}{g_w w_{diff}}$')
    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlim([-0.1, 1.1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')

    ax2.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax2.legend()

    fig12.savefig('./plot_producers/recall_noise_perfect_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

if recall_noise:
    # Network parameters
    w_diff_set = [3.0, 5.0, 8.0]
    successes_list = []
    persistent_time_list = []
    w_self = 1.0

    for w_diff in w_diff_set:
        w_rest = w_self - w_diff
        w_next = w_self - 0.5 * w_diff
        max_noise = 2 * g_w_ampa * w_diff

        w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)

        sigma_vector = np.linspace(0, max_noise, num=num)
        success_vector = np.zeros((num, trials))
        persistent_time_vector = np.zeros((num, trials))

        for index_sigma, sigma in enumerate(sigma_vector):
            for trial in range(trials):
                # Build the network
                nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                                  tau_m=tau_m,
                                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa,
                                  tau_p=tau_p,
                                  z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                                  perfect=perfect,
                                  k_perfect=k_perfect, always_learning=always_learning)
                # Build the manager
                manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
                nn.w_ampa = w

                # Recall
                T_recall = 3.0
                T_cue = 0.100
                sequences = [[i for i in range(n_patterns)]]
                n = 1

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector[index_sigma, trial] = success
                persistent_time_vector[index_sigma, trial] = mean

        successes_list.append(success_vector)
        persistent_time_list.append(persistent_time_vector)

    # Plot
    fig2 = plt.figure(figsize=figsize)
    ax = fig2.add_subplot(111)

    current_palette = sns.color_palette()

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):
        w_rest = w_self - w_diff
        max_noise = 2 * g_w_ampa * w_diff
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / (2 * g_w_ampa * w_diff)

        successes = success_vector.mean(axis=1)
        std = success_vector.std(axis=1)

        ax.plot(sigmas_norm, successes, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
                label=r'$w_{rest} = $' + str(w_rest))

        ax.fill_between(sigmas_norm, successes - std, successes + std,
                        color=current_palette[index], alpha=alpha)

    ax.set_ylabel('% Success')
    ax.set_xlabel(r'$\frac{\sigma}{2 g_w (w_{next} - w_{rest})}$')

    ax.set_xlim([-0.1, 1.1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax.legend()

    fig2.savefig('./plot_producers/recall_noise.pdf', frameon=False, dpi=110, bbox_inches='tight')

    fig22 = plt.figure(figsize=figsize)
    ax2 = fig22.add_subplot(111)

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):
        w_rest = w_self - w_diff
        max_noise = 2 * g_w_ampa * w_diff
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / (2 * g_w_ampa * w_diff)

        persistence_masked = np.ma.masked_less_equal(persistent_time_vector, 0.0)
        persistence = persistence_masked.mean(axis=1)
        std_pt = persistence_masked.std(axis=1)

        # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        std_pt.mask = accepted

        ax2.plot(sigmas_norm, persistence, 'o-', color=current_palette[index], markersize=markersize,
                 linewidth=linewidth,
                 label=r'$w_{rest} = $' + str(w_rest))

        ax2.fill_between(sigmas_norm, persistence - std_pt, persistence + std_pt,
                         color=current_palette[index], alpha=alpha)

    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlabel(r'$\frac{\sigma}{2 g_w (w_{next} - w_{rest})}$')
    ax2.set_xlim([-0.1, 1.1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')

    ax2.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax2.legend()

    fig22.savefig('./plot_producers/recall_noise_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

if matrix_noise:
    # Network parameters
    w_diff_set = [2.0, 4.0, 6.0]
    successes_list = []
    persistent_time_list = []
    w_self = 1.0

    for w_diff in w_diff_set:
        w_rest = w_self - w_diff
        w_next = w_self - 0.5 * w_diff
        max_noise = w_diff / 2.0

        w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)

        sigma_vector = np.linspace(0, max_noise, num=num)
        success_vector = np.zeros((num, trials))
        persistent_time_vector = np.zeros((num, trials))

        for index_sigma, sigma in enumerate(sigma_vector):
            for trial in range(trials):
                # Build the network
                nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                                  tau_m=tau_m,
                                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa,
                                  tau_p=tau_p,
                                  z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                                  perfect=perfect,
                                  k_perfect=k_perfect, always_learning=always_learning)

                # Build the manager
                manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
                matrix_noise = np.random.normal(loc=0, scale=sigma, size=w.shape)
                nn.w_ampa = w + matrix_noise

                # Recall
                T_recall = 3.0
                T_cue = 0.100
                sequences = [[i for i in range(n_patterns)]]
                n = 1

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector[index_sigma, trial] = success
                persistent_time_vector[index_sigma, trial] = mean

        successes_list.append(success_vector)
        persistent_time_list.append(persistent_time_vector)

    # Plot
    fig3 = plt.figure(figsize=figsize)
    ax = fig3.add_subplot(111)
    current_palette = sns.color_palette()

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):
        w_rest = w_self - w_diff
        max_noise = w_diff / 2.0
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / max_noise

        successes = success_vector.mean(axis=1)
        std = success_vector.std(axis=1)

        ax.plot(sigmas_norm, successes, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
                label=r'$w_{rest} = $' + str(w_rest))
        ax.fill_between(sigmas_norm, successes - std, successes + std, color=current_palette[index], alpha=alpha)

    ax.set_ylabel('% Success')
    ax.set_xlabel(r'$\frac{2 \sigma}{(w_{self} - w_{rest})}$')
    ax.set_xlim([-0.1, 1.1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax.legend()

    fig3.savefig('./plot_producers/matrix_noise.pdf', frameon=False, dpi=110, bbox_inches='tight')

    fig32 = plt.figure(figsize=figsize)
    ax2 = fig32.add_subplot(111)

    for index, (w_diff, success_vector, persistent_time_vector) in enumerate(
            zip(w_diff_set, successes_list, persistent_time_list)):
        w_rest = w_self - w_diff
        max_noise = w_diff / 2.0
        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / max_noise

        persistence_masked = np.ma.masked_less_equal(persistent_time_vector, 0.0)
        persistence = persistence_masked.mean(axis=1)
        std_pt = persistence_masked.std(axis=1)

        # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        std_pt.mask = accepted

        ax2.plot(sigmas_norm, persistence, 'o-', color=current_palette[index], markersize=markersize,
                 linewidth=linewidth,
                 label=r'$w_{rest} = $' + str(w_rest))

        ax2.fill_between(sigmas_norm, persistence - std_pt, persistence + std_pt,
                         color=current_palette[index], alpha=alpha)

    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlabel(r'$\frac{2 \sigma}{(w_{self} - w_{rest})}$')
    ax2.set_xlim([-0.1, 1.1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')

    ax2.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax2.legend()

    fig32.savefig('./plot_producers/matrix_noise_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

if minicolumns_recall_noise:
    # Network parameters
    minicolumns_set = [10, 20, 30]
    successes_list = []
    persistent_time_list = []

    w_self = 1.0
    w_next = -1.0
    w_rest = -3.0
    w_diff = w_self - w_rest
    max_noise = 2 * g_w_ampa * w_diff
    sigma_vector = np.linspace(0, max_noise, num=num)

    for minicolumns_ in minicolumns_set:
        success_vector = np.zeros((num, trials))
        persistent_time_vector = np.zeros((num, trials))
        w = simple_bcpnn_matrix(minicolumns_, w_self, w_next, w_rest)

        for index_sigma, sigma in enumerate(sigma_vector):
            for trial in range(trials):
                # Build the network
                nn = BCPNNPerfect(hypercolumns, minicolumns_, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                                  tau_m=tau_m,
                                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa,
                                  tau_p=tau_p,
                                  z_transfer=z_transfer, diagonal_zero=False, strict_maximum=strict_maximum,
                                  perfect=perfect,
                                  k_perfect=k_perfect, always_learning=always_learning)

                # Build the manager
                manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
                nn.w_ampa = w

                # Recall
                T_recall = 3.0
                T_cue = 0.100
                sequences = [[i for i in range(n_patterns)]]
                n = 1

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector[index_sigma, trial] = success
                persistent_time_vector[index_sigma, trial] = mean

        successes_list.append(success_vector)
        persistent_time_list.append(persistent_time_vector)

    # Plot
    fig4 = plt.figure(figsize=figsize)
    ax = fig4.add_subplot(111)

    current_palette = sns.color_palette()

    for index, (minicolumns_, success_vector, persistent_time_vector) in enumerate(zip(minicolumns_set, successes_list, persistent_time_list)):

        sigmas_norm = sigma_vector / max_noise
        successes = success_vector.mean(axis=1)
        std = success_vector.std(axis=1)

        ax.plot(sigmas_norm, successes, 'o-', color=current_palette[index], markersize=markersize,
                linewidth=linewidth, label=r'# units = ' + str(minicolumns_))
        ax.fill_between(sigmas_norm, successes - std, successes + std, color=current_palette[index], alpha=0.25)

    ax.set_ylabel('% Success')
    ax.set_xlabel(r'$\frac{\sigma}{2 g_w (w_{next} - w_{rest})}$')
    ax.set_xlim([-0.1, 1.1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax.legend()

    fig4.savefig('./plot_producers/minicolumns_recall_noise.pdf', frameon=False, dpi=110, bbox_inches='tight')

    fig42 = plt.figure(figsize=figsize)
    ax2 = fig42.add_subplot(111)

    for index, (minicolumns_, success_vector, persistent_time_vector) in enumerate(zip(minicolumns_set, successes_list, persistent_time_list)):
        sigmas_norm = sigma_vector / max_noise

        persistence_masked = np.ma.masked_less_equal(persistent_time_vector, 0.0)
        persistence = persistence_masked.mean(axis=1)
        std_pt = persistence_masked.std(axis=1)

        # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        std_pt.mask = accepted

        ax2.plot(sigmas_norm, persistence, 'o-', color=current_palette[index], markersize=markersize,
                 linewidth=linewidth, label=r'# units = ' + str(minicolumns_))

        ax2.fill_between(sigmas_norm, persistence - std_pt, persistence + std_pt,
                         color=current_palette[index], alpha=alpha)

    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlabel(r'$\frac{\sigma}{2 g_w (w_{next} - w_{rest})}$')
    ax2.set_xlim([-0.1, 1.1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')

    ax2.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax2.legend()

    fig42.savefig('./plot_producers/minicolumns_recall_noise_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

if minicolumns_matrix_noise:
    # Network parameters
    minicolumns_set = [10, 20, 30]
    successes_list = []
    persistent_time_list = []
    w_self = 1.0
    w_next = -1.0
    w_rest = -3.0

    w_diff = w_self - w_rest
    max_noise = w_diff / 2.0
    sigma_vector = np.linspace(0, max_noise, num=num)

    for minicolumns_ in minicolumns_set:

        w = simple_bcpnn_matrix(minicolumns_, w_self, w_next, w_rest)
        success_vector = np.zeros((num, trials))
        persistent_time_vector = np.zeros((num, trials))

        for index_sigma, sigma in enumerate(sigma_vector):
            for trial in range(trials):
                # Build the network
                nn = BCPNNPerfect(hypercolumns, minicolumns_, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                                  tau_m=tau_m,
                                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa,
                                  tau_p=tau_p,
                                  z_transfer=z_transfer, diagonal_zero=False, strict_maximum=strict_maximum,
                                  perfect=perfect,
                                  k_perfect=k_perfect, always_learning=always_learning)

                # Build the manager
                manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
                matrix_noise = np.random.normal(loc=0, scale=sigma, size=w.shape)
                nn.w_ampa = w + matrix_noise

                # Recall
                T_recall = 3.0
                T_cue = 0.100
                sequences = [[i for i in range(n_patterns)]]
                n = 1

                aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
                total_sequence_time, mean, std, success, timings = aux

                success_vector[index_sigma, trial] = success
                persistent_time_vector[index_sigma, trial] = mean

        successes_list.append(success_vector)
        persistent_time_list.append(persistent_time_vector)

    # Plot
    fig5 = plt.figure(figsize=figsize)
    ax = fig5.add_subplot(111)

    current_palette = sns.color_palette()

    for index, (minicolumns_, success_vector, persistent_time_vector) in enumerate(
            zip(minicolumns_set, successes_list, persistent_time_list)):

        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / max_noise

        successes = success_vector.mean(axis=1)
        std = success_vector.std(axis=1)

        ax.plot(sigmas_norm, successes, 'o-', color=current_palette[index], markersize=markersize, linewidth=linewidth,
                label=r'# units = ' + str(minicolumns_))
        ax.fill_between(sigmas_norm, successes - std, successes + std, color=current_palette[index], alpha=alpha)

    ax.set_ylabel('% Success')
    ax.set_xlabel(r'$\frac{2 \sigma}{(w_{self} - w_{rest})}$')
    ax.set_xlim([-0.1, 1.1])

    ax.axhline(0, ls='--', color='gray')
    ax.axvline(0, ls='--', color='gray')

    ax.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax.legend()

    fig5.savefig('./plot_producers/matrix_noise_minicolumns.pdf', frameon=False, dpi=110, bbox_inches='tight')

    fig52 = plt.figure(figsize=figsize)
    ax2 = fig52.add_subplot(111)

    for index, (minicolumns_, success_vector, persistent_time_vector) in enumerate(
            zip(minicolumns_set, successes_list, persistent_time_list)):

        sigma_vector = np.linspace(0, max_noise, num=num)
        sigmas_norm = sigma_vector / max_noise

        persistence_masked = np.ma.masked_less_equal(persistent_time_vector, 0.0)
        persistence = persistence_masked.mean(axis=1)
        std_pt = persistence_masked.std(axis=1)

        # Exclude percentage (This ecludes the indexes that fail to converge in failure_threshold % of the trials
        accepted = persistence_masked.mask.sum(axis=1) >= trials * failure_treshold
        persistence.mask = accepted
        std_pt.mask = accepted

        ax2.plot(sigmas_norm, persistence, 'o-', color=current_palette[index], markersize=markersize,
                 linewidth=linewidth,label=r'# units = ' + str(minicolumns_))

        ax2.fill_between(sigmas_norm, persistence - std_pt, persistence + std_pt,
                         color=current_palette[index], alpha=alpha)

    ax2.set_ylabel(r'$T_{persistence}$')
    ax2.set_xlabel(r'$\frac{2 \sigma}{(w_{self} - w_{rest})}$')
    ax2.set_xlim([-0.1, 1.1])

    ax2.axhline(0, ls='--', color='gray')
    ax2.axvline(0, ls='--', color='gray')

    ax2.set_title('Robustness of the system to noise, trials = ' + str(trials))
    ax2.legend()

    fig52.savefig('./plot_producers/matrix_noise_minicolumns_pt.pdf', frameon=False, dpi=110, bbox_inches='tight')

plt.close()