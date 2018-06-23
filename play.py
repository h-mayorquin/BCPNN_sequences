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

plt.rcParams['figure.figsize'] = (16, 12)

np.set_printoptions(suppress=True, precision=2)

sns.set(font_scale=3.0)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

plt.rcParams['figure.figsize'] = (12.9, 12)

np.set_printoptions(suppress=True, precision=5)

sns.set(font_scale=3.5)

from network import Protocol, NetworkManager, BCPNNPerfect, TimedInput
from connectivity_functions import create_orthogonal_canonical_representation, build_network_representation
from connectivity_functions import get_weights_from_probabilities, get_probabilities_from_network_representation
from connectivity_functions import create_matrix_from_sequences_representation, produce_overlaped_sequences
from analysis_functions import calculate_recall_time_quantities, get_weights
from analysis_functions import get_weights_collections
from plotting_functions import plot_network_activity_angle, plot_weight_matrix
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
from analysis_functions import calculate_patterns_timings, calculate_recall_success_nr

epsilon = 10e-60
if False:
    always_learning = False
    strict_maximum = True
    perfect = False
    z_transfer = False
    k_perfect = True
    diagonal_zero = False
    normalized_currents = True

    g_w_ampa = 1.0
    g_w = 0.0
    g_a = 1.0
    tau_a = 0.150
    G = 1.0
    sigma = 0.0
    tau_m = 0.020
    tau_z_pre_ampa = 0.050
    tau_z_post_ampa = 0.025
    tau_p = 10.0

    hypercolumns = 1
    minicolumns = 10
    n_patterns = 10

    # Manager properties
    dt = 0.001
    values_to_save = ['o', 's', 'i_ampa', 'a']

    # Protocol
    training_time = 0.100
    inter_sequence_interval = 0.0

    # Recall
    T_cue = 0.020
    T_recall = 1.0 + T_cue
    n = 1


    # Neural Network
    nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                      sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                      z_transfer=z_transfer, diagonal_zero=diagonal_zero, strict_maximum=strict_maximum,
                      perfect=perfect, k_perfect=k_perfect, always_learning=always_learning,
                      normalized_currents=normalized_currents)

    # Build the manager
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

    # Protocol
    matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)
    seq = np.copy(matrix)
    seq[4] = matrix[2]
    seq[5:] = matrix[4:-1]
    nr = build_network_representation(seq, minicolumns, hypercolumns)

    n_connections = len(seq) - 1
    value = 1.0
    extension = 10
    alpha = 1.0
    weights = [value for i in range(n_connections)]
    weights_collection = [weights]
    sequences = [seq]
    w = create_matrix_from_sequences_representation(minicolumns, hypercolumns, sequences, weights_collection,
                                                    extension, alpha, w_min=-10)
    pprint.pprint(seq)
    nn.w_ampa = w
    aux, indexes = np.unique(nr, axis=0, return_index=True)
    patterns_dic = {index:pattern for (index, pattern) in zip(indexes, aux)}
    manager.patterns_dic = patterns_dic

    plot_weight_matrix(nn, ampa=True)
    plt.show()
    plt.plot()


always_learning = False
strict_maximum = True
perfect = False
z_transfer = False
k_perfect = True
diagonal_zero = False
normalized_currents = True

g_w_ampa = 1.0
g_w = 0.0
g_a = 1.0
tau_a = 0.250
g_beta = 0.0
G = 1.0
sigma = 0.0
tau_m = 0.020
tau_z_pre_ampa = 0.025
tau_z_post_ampa = 0.025
tau_p = 10.0

hypercolumns = 1
minicolumns = 20
n_patterns = 20

# Manager properties
dt = 0.001
values_to_save = ['o', 's']

# Protocol
training_time = 0.100
inter_sequence_interval = 0.0
inter_pulse_interval = 0.0
epochs = 1
mixed_start = False
contiguous = True
remove = 0.010
s = 1.0
r = 0.25
extension = 15

# Recall
T_cue = 0.020
T_recall = 1.0
T_persistence = 0.050
T_recall = T_cue + (1.5) * T_persistence * n_patterns / 2.0
T_persistence = max(0.005, T_persistence - tau_m)

factor = 0.01

# Neural Network
nn = BCPNNPerfect(hypercolumns, minicolumns, g_w_ampa=g_w_ampa, g_w=g_w, g_a=g_a, tau_a=tau_a, tau_m=tau_m,
                  sigma=sigma, G=G, tau_z_pre_ampa=tau_z_pre_ampa, tau_z_post_ampa=tau_z_post_ampa, tau_p=tau_p,
                  g_beta=g_beta, z_transfer=z_transfer, diagonal_zero=diagonal_zero,
                  strict_maximum=strict_maximum, perfect=perfect, k_perfect=k_perfect,
                  always_learning=always_learning, normalized_currents=normalized_currents)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)

# Protocol
matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)
seq1, seq2 = produce_overlaped_sequences(minicolumns, hypercolumns, n_patterns, s, r,
                                         mixed_start=mixed_start, contiguous=contiguous)
n_connections = len(seq1) - 1
nr1 = build_network_representation(seq1, minicolumns, hypercolumns)
nr2 = build_network_representation(seq2, minicolumns, hypercolumns)

Bs = [1 - np.exp(-T_persistence / tau_a) for i in range(n_connections)]
value = 1.0
alpha = Bs[0]
weights = [value for B in Bs]
weights_collection = [weights, weights]
sequences = [seq1, seq2]
w = create_matrix_from_sequences_representation(minicolumns, hypercolumns, sequences, weights_collection,
                                                extension, alpha, w_min=0)

nr = np.concatenate((nr1, nr2))
aux, indexes = np.unique(nr, axis=0, return_index=True)
patterns_dic = {index:pattern for (index, pattern) in zip(indexes, aux)}
manager.patterns_dic = patterns_dic

nn.w_ampa = w
w_diff = 2 * alpha
current = 2  * g_w_ampa * w_diff
noise = factor * current
nn.sigma = noise

aux = calculate_recall_success_nr(manager, nr2, T_recall, T_cue, debug=True, remove=remove)
s, timings, pattern_sequence = aux
plot_network_activity_angle(manager)
plt.show()