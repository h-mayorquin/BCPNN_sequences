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

np.set_printoptions(suppress=True, precision=2)

sns.set(font_scale=3.5)
sns.set_style(style='white')

from network import Protocol, BCPNNModular, NetworkManager, BCPNNPerfect
from plotting_functions import plot_weight_matrix, plot_state_variables_vs_time, plot_winning_pattern
from plotting_functions import plot_network_activity, plot_network_activity_angle
from analysis_functions import calculate_recall_time_quantities, calculate_angle_from_history
from connectivity_functions import artificial_connectivity_matrix

def simple_bcpnn_theo_recall_time(tau_a, g_a, g_w, w_next, w_self):

    delta_w = w_self - w_next
    return tau_a * np.log(g_a / (g_a - g_w * delta_w))

def simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest):

    w = np.ones((minicolumns, minicolumns)) * w_rest
    for i in range(minicolumns):
        w[i, i] = w_self

    for i in range(minicolumns -1):
        w[i + 1, i] = w_next

    return w

#######
# General parameters
#######
g_w_ampa = 2.0
g_w = 0.0
g_a = 10.0
tau_a = 0.250
tau_z = 0.150
G = 1.0
sigma = 0.0

w_self = 1.0
w_next = -0.1
w_rest = -0.2


markersize = 32
linewdith = 10

#######
# tau a
#######

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'i_ampa', 'i_nmda']

# Protocol
training_time = 0.100
inter_sequence_interval = 1.0
inter_pulse_interval = 0.0

tau_a_vector = np.linspace(0.050, 1.0, 10)
T_recall_vector_tau_a = np.zeros_like(tau_a_vector)
success_tau_a = np.zeros_like(tau_a_vector)
std_tau_a = np.zeros_like(tau_a_vector)

for index, tau_a_ in enumerate(tau_a_vector):

    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a_,
                      sigma=sigma, G=G,
                      z_transfer=False, diagonal_zero=False, strict_maximum=True, perfect=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
    w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)
    nn.w_ampa = w

    # Recall
    T_recall = 1.0
    T_cue = 0.100
    sequences = [[i for i in range(n_patterns)]]
    n = 1

    aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
    total_sequence_time, mean, std, success, timings = aux

    T_recall_vector_tau_a[index] = mean
    success_tau_a[index] = success
    std_tau_a[index] = std


T_recall_theorical_tau_a = simple_bcpnn_theo_recall_time(tau_a_vector, g_a, g_w_ampa, w_next, w_self)

fig1 = plt.figure(figsize=(16, 12))
ax1 = fig1.add_subplot(111)
ax1.plot(tau_a_vector, T_recall_theorical_tau_a, '-', lw=linewdith, label=r'Theory')
ax1.plot(tau_a_vector, T_recall_vector_tau_a, 'o', markersize=markersize, label=r'Simulation')

ax1.axhline(0, ls='--', color='black')
ax1.axvline(0, ls='--', color='black')
ax1.set_xlabel(r'$\tau_{a}$ (s)')
ax1.set_ylabel(r'$T_{persistence}$ (s)')
ax1.legend()

fig1.savefig('./plot_producers/simple_bcpnn_tau_a.pdf', frameon=False, dpi=110, bbox_inches='tight')

###########
# g_a
###########
# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'i_ampa', 'i_nmda']

g_a_vector = g_w_ampa * (w_self - w_next) + np.logspace(0, 1.2, 20) * 0.5
T_recall_vector_g_a = np.zeros_like(g_a_vector)
success_g_a = np.zeros_like(g_a_vector)
std_g_a = np.zeros_like(g_a_vector)

for index, g_a_ in enumerate(g_a_vector):
    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a_, tau_a=tau_a,
                      sigma=sigma, G=G,
                      z_transfer=False, diagonal_zero=False, strict_maximum=True, perfect=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
    w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)
    nn.w_ampa = w

    # Recall
    T_recall = 4.0
    T_cue = 0.100
    sequences = [[i for i in range(n_patterns)]]
    n = 1

    aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
    total_sequence_time, mean, std, success, timings = aux

    T_recall_vector_g_a[index] = mean
    success_g_a[index] = success
    std_g_a[index] = std

T_recall_theoretical_g_a = simple_bcpnn_theo_recall_time(tau_a, g_a_vector, g_w_ampa, w_next, w_self)
singularity = g_w_ampa * (w_self - w_next)

fig2 = plt.figure(figsize=(16, 12))
ax2 = fig2.add_subplot(111)

ax2.plot(g_a_vector, T_recall_theoretical_g_a, '-',  lw=linewdith, label=r'Theory')
ax2.plot(g_a_vector, T_recall_vector_g_a, 'o', markersize=markersize,  label=r'Simulation')
ax2.axhline(0, ls='--', color='black')
ax2.axvline(0, ls='--', color='black')
ax2.axvline(singularity, ls='-', color='red', label=r'$g_{w}(w_{self} - w_{next})$')

ax2.set_xlabel(r'$g_{a}$')
ax2.set_ylabel(r'$T_{persistence}$ (s)')
ax2.legend();

fig2.savefig('./plot_producers/simple_bcpnn_g_a.pdf', frameon=False, dpi=110, bbox_inches='tight')

############
# g_w
############
# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'i_ampa', 'i_nmda']

g_w_max = g_a / (w_self - w_next)
g_w_ampa_vector = g_w_max - np.logspace(-1, 1, 20) * 0.9
T_recall_vector_g_w = np.zeros_like(g_w_ampa_vector)
success_g_w = np.zeros_like(g_w_ampa_vector)
std_g_w = np.zeros_like(g_w_ampa_vector)

for index, g_w_ampa_ in enumerate(g_w_ampa_vector):

    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa_, g_w=g_w, g_a=g_a, tau_a=tau_a,
                      sigma=sigma, G=G,
                      z_transfer=False, diagonal_zero=False, strict_maximum=True, perfect=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
    w = simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest)
    nn.w_ampa = w

    # Recall
    T_recall = 5.0
    T_cue = 0.100
    sequences = [[i for i in range(n_patterns)]]
    n = 1

    aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
    total_sequence_time, mean, std, success, timings = aux

    T_recall_vector_g_w[index] = mean
    success_g_w[index] = success
    std_g_w[index] = std

T_recall_theorical_g_w_ampa = simple_bcpnn_theo_recall_time(tau_a, g_a, g_w_ampa_vector, w_next, w_self)

fig3 = plt.figure(figsize=(16, 12))

ax3 = fig3.add_subplot(111)
ax3.plot(g_w_ampa_vector, T_recall_theorical_g_w_ampa, '-', lw=linewdith, label=r'Theory')
ax3.plot(g_w_ampa_vector, T_recall_vector_g_w, 'o', markersize=markersize, label=r'Simulation')
ax3.axhline(0, ls='--', color='black')
ax3.axvline(0, ls='--', color='black')
ax3.axvline(g_w_max, ls='-', color='red', label=r'$\frac{g_{a}}{(w_{self} - w_{next})}$')
ax3.set_xlabel(r'$g_{w}$')
ax3.set_ylabel(r'$T_{persistence}$ (s)')
ax3.legend();

fig3.savefig('./plot_producers/simple_bcpnn_g_w.pdf', frameon=False, dpi=110, bbox_inches='tight')

###########
# w_next
###########

# Patterns parameters
hypercolumns = 1
minicolumns = 10
n_patterns = 10

# Manager properties
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'i_ampa', 'i_nmda']

w_next_max = w_self
w_next_min = w_self - g_a / g_w_ampa
w_next_vector = (w_next_max - w_next_min) * np.logspace(-2, 0, 20) + w_next_min
T_recall_vector_w_next = np.zeros_like(w_next_vector)
success_w_next = np.zeros_like(w_next_vector)
std_w_next = np.zeros_like(w_next_vector)

for index, w_next_ in enumerate(w_next_vector):
    # Build the network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a,
                      sigma=sigma, G=G,
                      z_transfer=False, diagonal_zero=False, strict_maximum=True, perfect=True)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
    w = simple_bcpnn_matrix(minicolumns, w_self, w_next_, w_next_min - 1.0)
    nn.w_ampa = w

    # Recall
    T_recall = 5.0
    T_cue = 0.100
    sequences = [[i for i in range(n_patterns)]]
    n = 1

    aux = calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences)
    total_sequence_time, mean, std, success, timings = aux

    T_recall_vector_w_next[index] = mean
    success_w_next[index] = success
    std_w_next[index] = std

T_recall_theorical_w_next = simple_bcpnn_theo_recall_time(tau_a, g_a, g_w_ampa, w_next_vector, w_self)

fig4 = plt.figure(figsize=(16, 12))

ax4 = fig4.add_subplot(111)
ax4.plot(w_next_vector, T_recall_theorical_w_next, '-', lw=linewdith, label=r'Theory')
ax4.plot(w_next_vector, T_recall_vector_w_next, 'o', markersize=markersize, label=r'Simulation')

ax4.axhline(0, ls='--', color='black')
ax4.axvline(0, ls='--', color='black')
ax4.set_xlabel(r'$w_{next}$')
ax4.set_ylabel(r'$T_{persistence}$ (s)')
ax4.axvline(w_next_max, ls='-', color='red', label=r'$w_{self}$')
ax4.axvline(w_next_min, ls='-', color='red', label=r'$w_{self} - \frac{g_a}{g_w}$')
ax4.legend()

fig4.savefig('./plot_producers/simple_bcpnn_w_next.pdf', frameon=False, dpi=110, bbox_inches='tight')


plt.close()