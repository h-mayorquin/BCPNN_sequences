import random
import numpy as np
from connectivity_functions import softmax, get_w_pre_post, get_beta, log_epsilon
from data_transformer import build_ortogonal_patterns
import IPython


epoch_end_string = 'epoch_end'


class BCPNNModular:
    def __init__(self, hypercolumns, minicolumns, beta=None, w=None, G=1.0, tau_m=0.020, g_w=1.0, g_w_ampa=1.0, g_beta=1,
                 tau_z_pre=0.150, tau_z_post=0.005, tau_z_pre_ampa=0.005, tau_z_post_ampa=0.005, tau_p=10.0, tau_k=0.010,
                 tau_a=2.70, g_a=97.0, g_I=10.0, p=1.0, k=0.0, sigma=1.0, epsilon=1e-20, k_inner=False, prng=np.random):
        # Initial values are taken from the paper on memory by Marklund and Lansner also from Phil's paper

        # Random number generator
        self.prng = prng
        self.sigma = sigma
        self.epsilon = epsilon

        # Network parameters
        self.hypercolumns = hypercolumns
        self.minicolumns = minicolumns

        self.n_units = self.hypercolumns * self.minicolumns

        if hypercolumns == 1 and g_w < self.epsilon:
            self.diagonal_zero = False
        else:
            self.diagonal_zero = True

        # Connectivity
        self.beta = beta
        self.w = w

        #  Dynamic Parameters
        self.G = G
        self.tau_m = tau_m
        self.tau_z_pre = tau_z_pre
        self.tau_z_post = tau_z_post
        self.tau_z_pre_ampa = tau_z_pre_ampa
        self.tau_z_post_ampa = tau_z_post_ampa
        self.tau_p = tau_p
        self.tau_a = tau_a
        self.g_a = g_a
        self.g_w = g_w
        self.g_w_ampa = g_w_ampa
        self.g_beta = g_beta
        self.g_I = g_I

        self.k = k
        self.tau_k = tau_k
        self.k_d = 0
        self.k_inner = k_inner

        self.p = p

        # State variables
        self.o = np.zeros(self.n_units) * (1.0 / self.minicolumns)
        self.s = np.log(np.ones(self.n_units) * (1.0 / self.minicolumns))
        self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        # NMDA values
        self.i_nmda = np.zeros(self.n_units)
        self.z_pre = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_post = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.p_pre = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_post = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.w = np.zeros((self.n_units, self.n_units))

        # Ampa values
        self.i_ampa = np.zeros(self.n_units)
        self.z_pre_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_post_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_co_ampa = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.p_pre_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_post_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_co_ampa = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.w_ampa = np.zeros((self.n_units, self.n_units))

        # Set the adaptation to zeros by default
        self.a = np.zeros_like(self.o)
        # Set the clamping to zero by default
        self.I = np.zeros_like(self.o)

    def get_parameters(self):
        """
        Get the parameters of the model

        :return: a dictionary with the parameters
        """
        parameters = {'tau_m': self.tau_m, 'tau_z_post': self.tau_z_post, 'tau_z_pre': self.tau_z_pre,
                      'tau_p': self.tau_p, 'tau_a': self.tau_a, 'g_a': self.g_a, 'g_w': self.g_w,
                      'g_beta': self.g_beta, 'g_I':self.g_I, 'sigma':self.sigma, 'k': self.k,
                      'g_w_ampa': self.g_w_ampa, 'tau_z_post_ampa': self.tau_z_post_ampa,
                      'tau_z_pre_ampa': self.tau_z_pre_ampa, 'epsilon': self.epsilon, 'G': self.G}

        return parameters

    def reset_values(self, keep_connectivity=True):
        # State variables
        self.o = np.zeros(self.n_units) * (1.0 / self.minicolumns)
        self.s = np.log(np.ones(self.n_units) * (1.0 / self.minicolumns))
        self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))

        # NMDA values
        self.i_nmda = np.zeros(self.n_units)
        self.z_pre = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_post = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.p_pre = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_post = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_co = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)

        # Ampa values
        self.i_ampa = np.zeros(self.n_units)
        self.z_pre_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_post_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.z_co_ampa = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)
        self.p_pre_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_post_ampa = np.ones(self.n_units) * 1.0 / self.minicolumns
        self.p_co_ampa = np.ones((self.n_units, self.n_units)) * 1.0 / (self.minicolumns ** 2)


        # Set the adaptation to zeros by default
        self.a = np.zeros_like(self.o)
        # Set the clamping to zero by default
        self.I = np.zeros_like(self.o)

        if not keep_connectivity:
            self.beta = np.log(np.ones_like(self.o) * (1.0 / self.minicolumns))
            self.w = np.zeros((self.n_units, self.n_units))
            self.w_ampa = np.zeros((self.n_units, self.n_units))

    def randomize_pattern(self):
        self.o = self.prng.rand(self.n_units)
        self.s = np.log(self.prng.rand(self.n_units))

        self.z_pre = self.prng.rand(self.n_units)
        self.z_pre_ampa = self.prng.rand(self.n_units)

        # A follows, if o is randomized sent a to zero.
        self.a = np.zeros_like(self.o)

    def update_continuous(self, dt=1.0, sigma=None):

        if sigma is None:
            sigma = self.prng.normal(0, self.sigma, self.n_units)

        # Updated the probability and the support
        self.i_nmda = self.g_w * self.w @ self.z_pre
        self.i_ampa = self.g_w_ampa * self.w_ampa @ self.z_pre_ampa

        self.s += (dt / self.tau_m) * (self.i_nmda  # NMDA effects
                                       + self.i_ampa  # Ampa effects
                                       + self.g_beta * self.beta  # Bias
                                       + self.g_I * log_epsilon(self.I)  # Input current
                                       - self.s  # s follow all of the s above
                                       - self.g_a * self.a  # Adaptation
                                       + sigma)  # This last term is the noise
        # Soft-max
        self.o = softmax(self.s, t=self.G, minicolumns=self.minicolumns)

        # Update the adaptation
        self.a += (dt / self.tau_a) * (self.o - self.a)

        # Updated the z-traces
        self.z_pre += (dt / self.tau_z_pre) * (self.o - self.z_pre)
        self.z_post += (dt / self.tau_z_post) * (self.o - self.z_post)
        self.z_co = np.outer(self.z_post, self.z_pre)

        # Updated the z-traces AMPA
        self.z_pre_ampa += (dt / self.tau_z_pre_ampa) * (self.o - self.z_pre_ampa)
        self.z_post_ampa += (dt / self.tau_z_post_ampa) * (self.o - self.z_post_ampa)
        self.z_co_ampa = np.outer(self.z_post_ampa, self.z_pre_ampa)

        # Modulatory variables
        self.p += (dt / self.tau_p) * (1 - self.p)

        if self.k_inner:
            self.k_d += (dt / self.tau_k) * (self.k - self.k_d)

            # Updated the probability of the NMDA connection
            self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre) * self.k_d
            self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post) * self.k_d
            self.p_co += (dt / self.tau_p) * (self.z_co - self.p_co) * self.k_d

            # Updated the probability of AMPA connection
            self.p_pre_ampa += (dt / self.tau_p) * (self.z_pre_ampa - self.p_pre_ampa) * self.k_d
            self.p_post_ampa += (dt / self.tau_p) * (self.z_post_ampa - self.p_post_ampa) * self.k_d
            self.p_co_ampa += (dt / self.tau_p) * (self.z_co_ampa - self.p_co_ampa) * self.k_d

        else:
            # Updated the probability of the NMDA connection
            self.p_pre += (dt / self.tau_p) * (self.z_pre - self.p_pre)
            self.p_post += (dt / self.tau_p) * (self.z_post - self.p_post)
            self.p_co += (dt / self.tau_p) * (self.z_co - self.p_co)

            # Updated the probability of the AMPA connection
            self.p_pre_ampa += (dt / self.tau_p) * (self.z_pre_ampa - self.p_pre_ampa)
            self.p_post_ampa += (dt / self.tau_p) * (self.z_post_ampa - self.p_post_ampa)
            self.p_co_ampa += (dt / self.tau_p) * (self.z_co_ampa - self.p_co_ampa)

        if self.k > self.epsilon:
            self.beta = get_beta(self.p_post, self.epsilon)
            self.w_ampa = get_w_pre_post(self.p_co_ampa, self.p_pre_ampa, self.p_post_ampa, self.p,
                                         self.epsilon, diagonal_zero=self.diagonal_zero)
            self.w = get_w_pre_post(self.p_co, self.p_pre, self.p_post, self.p,
                                    self.epsilon, diagonal_zero=self.diagonal_zero)


class NetworkManager:
    """
    This class will run the BCPNN Network. Everything from running, saving and calculating quantities should be
    methods in this class.  In short this will do the running of the network, the learning protocols, etcera.

    Note that data analysis should be conducted into another class preferably.
    """

    def __init__(self, nn=None, dt=0.001, values_to_save=[]):
        """
        :param nn: A BCPNN instance
        :param time: A numopy array with the time to run
        :param values_to_save: a list with the values as strings of the state variables that should be saved
        """

        self.nn = nn

        # Timing variables
        self.dt = dt
        self.T_total = 0  # For plotting

        self.sampling_rate = 1.0

        # Initialize saving dictionary
        self.saving_dictionary = self.get_saving_dictionary(values_to_save)

        # Initialize the history dictionary for saving values
        self.history = None
        self.empty_history()

        # Trained patterns
        self.n_patterns = 0
        self.patterns_dic = build_ortogonal_patterns(self.nn.hypercolumns, self.nn.minicolumns)
        self.patterns = list(self.patterns_dic)
        self.stored_patterns_indexes = []

    def get_saving_dictionary(self, values_to_save):
        """
        This resets the saving dictionary and only activates the values in values_to_save
        """

        # Reinitialize the dictionary
        saving_dictionary = {'o': False, 's': False, 'a': False,
                             'z_pre': False, 'z_post': False, 'z_co': False,
                             'p_pre': False, 'p_post': False, 'p_co': False,
                             'z_pre_ampa': False,'z_post_ampa': False, 'z_co_ampa': False,
                             'p_pre_ampa': False, 'p_post_ampa': False, 'p_co_ampa': False,
                             'i_ampa': False, 'i_nmda': False,
                             'w_ampa': False, 'w': False, 'beta': False, 'p': False, 'k_d': False}

        # Activate the values passed to the function
        for state_variable in values_to_save:
            saving_dictionary[state_variable] = True

        return saving_dictionary

    def empty_history(self):
        """
        A function to empty the history
        """
        empty_array = np.array([]).reshape(0, self.nn.n_units)
        empty_array_square = np.array([]).reshape(0, self.nn.n_units, self.nn.n_units)

        self.history = {'o': empty_array, 's': empty_array, 'a': empty_array,
                        'z_pre': empty_array, 'z_post': empty_array,
                        'p_pre': empty_array, 'p_post': empty_array,
                        'z_co': empty_array_square, 'p_co': empty_array_square, 'w': empty_array_square,
                        'z_pre_ampa': empty_array, 'z_post_ampa': empty_array,
                        'p_pre_ampa': empty_array, 'p_post_ampa': empty_array,
                        'z_co_ampa': empty_array_square, 'p_co_ampa': empty_array_square, 'w_ampa': empty_array_square,
                        'i_nmda': empty_array, 'i_ampa': empty_array,
                        'beta': empty_array, 'k_d': np.array(([])), 'p': np.array([])}

    def append_history(self, history, saving_dictionary):
        """
        This function is used at every step of a process that is going to be saved. The items given by
        saving dictinoary will be appended to the elements of the history dictionary.

        :param history: is the dictionary with the saved values
        :param saving_dictionary:  a saving dictionary with keys as the parameters that should be saved
        and items as boolean indicating whether that parameters should be saved or not
        """

        # Dynamical variables
        if saving_dictionary['o']:
            history['o'].append(np.copy(self.nn.o))
        if saving_dictionary['s']:
            history['s'].append(np.copy(self.nn.s))
        if saving_dictionary['a']:
            history['a'].append(np.copy(self.nn.a))
        if saving_dictionary['k_d']:
            history['k_d'].append(np.copy(self.nn.k_d))
        if saving_dictionary['p']:
            history['p'].append(np.copy(self.nn.p))


        # NMDA connectivity
        if saving_dictionary['i_nmda']:
            history['i_nmda'].append(np.copy(self.nn.i_nmda))
        if saving_dictionary['z_pre']:
            history['z_pre'].append(np.copy(self.nn.z_pre))
        if saving_dictionary['z_post']:
            history['z_post'].append(np.copy(self.nn.z_post))
        if saving_dictionary['z_co']:
            history['z_co'].append(np.copy(self.nn.z_co))
        if saving_dictionary['p_pre']:
            history['p_pre'].append(np.copy(self.nn.p_pre))
        if saving_dictionary['p_post']:
            history['p_post'].append(np.copy(self.nn.p_post))
        if saving_dictionary['p_co']:
            history['p_co'].append(np.copy(self.nn.p_co))
        if saving_dictionary['w']:
            history['w'].append(np.copy(self.nn.w))

        # AMPA connectivity
        if saving_dictionary['i_ampa']:
            history['i_ampa'].append(np.copy(self.nn.i_ampa))
        if saving_dictionary['z_pre_ampa']:
            history['z_pre_ampa'].append(np.copy(self.nn.z_pre_ampa))
        if saving_dictionary['z_post_ampa']:
            history['z_post_ampa'].append(np.copy(self.nn.z_post_ampa))
        if saving_dictionary['z_co_ampa']:
            history['z_co_ampa'].append(np.copy(self.nn.z_co_ampa))
        if saving_dictionary['p_pre_ampa']:
            history['p_pre_ampa'].append(np.copy(self.nn.p_pre_ampa))
        if saving_dictionary['p_post_ampa']:
            history['p_post_ampa'].append(np.copy(self.nn.p_post_ampa))
        if saving_dictionary['p_co_ampa']:
            history['p_co_ampa'].append(np.copy(self.nn.p_co_ampa))
        if saving_dictionary['w_ampa']:
            history['w_ampa'].append(np.copy(self.nn.w_ampa))

        # Beta
        if saving_dictionary['beta']:
            history['beta'].append(np.copy(self.nn.beta))

    def run_network(self, time=None, I=None):
        # Change the time if given

        if time is None:
            raise ValueError('time has to be given')

        self.dt = time[1] - time[0]

        # Load the clamping if available
        if I is None:
            self.nn.I = np.zeros_like(self.nn.o)
        else:
            self.nn.I = self.patterns_dic[I]

        # Create a vector of noise
        if self.nn.sigma < self.nn.epsilon:
            noise = np.zeros((time.size, self.nn.n_units))
        else:
            noise = self.nn.prng.normal(0, self.nn.sigma, size=(time.size, self.nn.n_units))

        # Initialize run history
        step_history = {}

        # Create a list for the values that are in the saving dictionary
        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                step_history[quantity] = []

        # Run the simulation and save the values
        for index_t, t in enumerate(time):
            # Append the history first
            self.append_history(step_history, self.saving_dictionary)
            # Update the system with one step
            self.nn.update_continuous(dt=self.dt, sigma=noise[index_t, :])

        # Concatenate with the past history and redefine dictionary
        for quantity, boolean in self.saving_dictionary.items():
            if boolean:
                self.history[quantity] = np.concatenate((self.history[quantity], step_history[quantity]))

        return self.history

    def run_network_protocol(self, protocol, verbose=True, values_to_save_epoch=None, reset=True, empty_history=True):

        if empty_history:
            self.empty_history()
            self.T_total = 0
        if reset:
            self.nn.reset_values(keep_connectivity=True)

        # Unpack the protocol
        times = protocol.times_sequence
        patterns_sequence = protocol.patterns_sequence
        learning_constants = protocol.learning_constants_sequence  # The values of Kappa

        # Update list of stored patterns
        self.stored_patterns_indexes += []
        self.stored_patterns_indexes += protocol.patterns_indexes

        # This eliminates duplicates
        self.stored_patterns_indexes = list(set(self.stored_patterns_indexes))
        self.n_patterns = len(self.stored_patterns_indexes)

        total_time = 0

        epoch_history = {}
        # Initialize dictionary for storage

        if values_to_save_epoch:
            saving_dictionary_epoch = self.get_saving_dictionary(values_to_save_epoch)
            # Create a list for the values that are in the saving dictionary
            for quantity, boolean in saving_dictionary_epoch.items():
                if boolean:
                    epoch_history[quantity] = []

        # Run the protocol
        epochs = 0
        for time, pattern_index, k in zip(times, patterns_sequence, learning_constants):

            # End of the epoch
            if pattern_index == epoch_end_string:
                # Store the values at the end of the epoch
                if values_to_save_epoch:
                    self.append_history(epoch_history, saving_dictionary_epoch)

                if verbose:
                    print('epochs', epochs)
                    epochs += 1

            # Running step
            else:
                self.nn.k = k
                running_time = np.arange(0, time, self.dt)
                self.run_network(time=running_time, I=pattern_index)
                total_time += time

        # Record the total time
        self.T_total += total_time

        # Return the history if available
        if values_to_save_epoch:
            return epoch_history

    def run_network_recall(self, T_recall=10.0, T_cue=0.0, I_cue=None, reset=True, empty_history=True):
        """
        Run network free recall
        :param T_recall: The total time of recalling
        :param T_cue: the time that the cue is run
        :param I_cue: The current to give as the cue
        :param reset: Whether the state variables values should be returned
        :param empty_history: whether the history should be cleaned
        """
        self.nn.k = 0
        time_recalling = np.arange(0, T_recall, self.dt)
        time_cue = np.arange(0, T_cue, self.dt)

        if empty_history:
            self.empty_history()
            self.T_total = 0
        if reset:
            self.nn.reset_values(keep_connectivity=True)

        # Run the cue
        if T_cue > 0.001:
            self.run_network(time=time_cue, I=I_cue)

        # Run the recall
        self.run_network(time=time_recalling)

        # Calculate total time
        self.T_total += T_recall + T_cue


class Protocol:

    def __init__(self):

        self.patterns_indexes = None
        self.patterns_sequence = None
        self.times_sequence = None
        self.learning_constants_sequence = None
        self.epochs = None

    def simple_protocol(self, patterns_indexes, training_time=1.0, inter_pulse_interval=0.0,
                        inter_sequence_interval=1.0, epochs=1):
        """
        The simples protocol to train a sequence

        :param patterns_indexes: All the indexes patterns that will be train
        :param training_time: How long to present the pattern
        :param inter_pulse_interval: Time between each pattern
        :param inter_sequence_interval: Time between each repetition of the sequence
        :param epochs: how many times to present the sequence
        """

        epsilon = 1e-10
        self.epochs = epochs
        self.patterns_indexes = patterns_indexes

        patterns_sequence = []
        times_sequence = []
        learning_constants_sequence = []

        for i in range(epochs):
            # Let's fill the times
            for pattern in patterns_indexes:
                # This is when the pattern is training
                patterns_sequence.append(pattern)
                times_sequence.append(training_time)
                learning_constants_sequence.append(1.0)

                # This happens when there is time between the patterns
                if inter_pulse_interval > epsilon:
                    patterns_sequence.append(None)
                    times_sequence.append(inter_pulse_interval)
                    learning_constants_sequence.append(0.0)

            # Remove the inter pulse interval at the end of the patterns
            if inter_pulse_interval > epsilon:
                patterns_sequence.pop()
                times_sequence.pop()
                learning_constants_sequence.pop()

            if inter_sequence_interval > epsilon:
                patterns_sequence.append(None)
                times_sequence.append(inter_sequence_interval)
                learning_constants_sequence.append(0.0)

            # End of epoch
            if epochs > 1:
                patterns_sequence.append(epoch_end_string)
                times_sequence.append(epoch_end_string)
                learning_constants_sequence.append(epoch_end_string)

        # Store
        self.patterns_sequence = patterns_sequence
        self.times_sequence = times_sequence
        self.learning_constants_sequence = learning_constants_sequence

    def cross_protocol(self, chain, training_time=1.0,  inter_sequence_interval=1.0, epochs=1):

        self.epochs = epochs
        self.patterns_indexes = {pattern for patterns in chain for pattern in patterns}  # Neat double iteration
        self.patterns_indexes = list(self.patterns_indexes)
        print(self.patterns_indexes)

        patterns_sequence = []
        times_sequence = []
        learning_constant_sequence = []

        for i in range(epochs):
            for patterns in chain:
                # Get the chains one by one
                for pattern in patterns:
                    patterns_sequence.append(pattern)
                    times_sequence.append(training_time)
                    learning_constant_sequence.append(1.0)

                # Get a space between the chains
                patterns_sequence.append(None)
                times_sequence.append(inter_sequence_interval)
                learning_constant_sequence.append(0.0)

            # Get the epoch if necessary
            if epochs > 1:
                patterns_sequence.append(epoch_end_string)
                times_sequence.append(epoch_end_string)
                learning_constant_sequence.append(epoch_end_string)

        # Store
        self.patterns_sequence = patterns_sequence
        self.times_sequence = times_sequence
        self.learning_constants_sequence = learning_constant_sequence


    def create_overload_chain(self, number_of_sequences, half_width, units_to_overload):

        chain = []
        number = 0
        for dummy_index in range(number_of_sequences):

            sequence = []

            # The first half
            i = 0
            while i < half_width:
                if number in units_to_overload:
                    number += 1

                else:
                    sequence.append(number)
                    number += 1
                    i += 1

            # The overload units in the middle
            sequence += units_to_overload

            # The second half
            i = 0
            while i < half_width:
                if number in units_to_overload:
                    number += 1
                else:
                    sequence.append(number)
                    number += 1
                    i += 1

            chain.append(sequence)

        return chain

    def generate_sample_sequence(self, m, k, p, o, desired_sequences, order=False,
                                 array_view=False, overlap_view=False, verbose=False, seed=0, tolerance=100000):
        sequences = []
        p_array = np.zeros(m, dtype='int')
        numbers = {i for i in range(m)}
        random.seed(a=seed)
        n_sequences = 0
        count = 0

        while n_sequences < desired_sequences:
            count += 1
            if count > tolerance:
                print('tolerance breanched')
                print(tolerance)
                break

            sample = random.sample(numbers, k)
            # Routine that checks overlap
            o_flag = True
            for sequence in sequences:
                sample_set = set(sample)
                intersection = len(sample_set.intersection(sequence))
                if verbose:
                    print('check overlap')
                    print(sample_set)
                    print(sequence)
                    print(intersection)

                if intersection >= o:
                    o_flag = False

            # Routine that checks overload
            p_flag = True
            for element in sample:
                if p_array[element] >= p:
                    p_flag = False

            if verbose:
                print(sample)
                print('overlaps')
                print('p', p_flag)
                print('o', o_flag)
                print(sequences)
                print('---------------')

            # Add to the list of sequences and modify p_array
            if p_flag and o_flag:
                sequences.append(sample)
                n_sequences += 1

                # Modify p-array
                for element in sample:
                    p_array[element] += 1

                if verbose:
                    print('check p_array update')
                    print(sequences)
                    print(p_array)

        if order:
            for sequence in sequences:
                sequence.sort()

        if array_view:
            sequence_array = np.zeros((len(sequences), m))
            for index, sequence in enumerate(sequences):
                sequence_array[index, sequence] = 1

        if overlap_view:
            overlap_array = np.zeros((len(sequences), len(sequences)))
            for index_1, sequence1 in enumerate(sequence_array):
                for index_2, sequence2 in enumerate(sequence_array):
                    overlap_array[index_1, index_2] = np.dot(sequence1, sequence2)

        return sequences, p_array, sequence_array, overlap_array




