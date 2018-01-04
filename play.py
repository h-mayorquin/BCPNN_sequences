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

from network import Protocol, BCPNNFast, NetworkManager
from plotting_functions import plot_weight_matrix

# Patterns parameters
hypercolumns = 4
minicolumns = 20
dt = 0.001
values_to_save = ['o', 's', 'z_pre', 'z_post', 'a', 'p_pre', 'p_post', 'p_co', 'z_co', 'w', 'p',]

# Build the network
nn = BCPNNFast(hypercolumns, minicolumns)

# Build the manager
manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)


# Protocol
training_time = 0.1
inter_sequence_interval = 1.0
epochs = 3
number_of_sequences = 2
half_width = 3
units_to_overload = [0, 1]

# Build chain protocol
chain_protocol = Protocol()
sequences = chain_protocol.create_overload_chain(number_of_sequences, half_width, units_to_overload)
chain_protocol.cross_protocol(chain=sequences, training_time=training_time,
                              inter_sequence_interval=inter_sequence_interval, epochs=epochs)

# Train
epoch_history = manager.run_network_protocol(protocol=chain_protocol, verbose=True)