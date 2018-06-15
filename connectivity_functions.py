import numpy as np
import IPython

def log_epsilon(x, epsilon=1e-10):

    return np.log10(np.maximum(x, epsilon))


def calculate_probability(patterns):
    """
    Returns the probability from a list of patterns to be learned
    :param patterns: list of patterns to be learned
    :return:
    """

    p = np.zeros(patterns[0].size)
    number_of_patterns = len(patterns)

    for pattern in patterns:
        p += pattern

    p /= number_of_patterns

    return p


def calculate_coactivations(patterns):

    coactivations = np.zeros((patterns[0].size, patterns[0].size))
    number_of_patterns = len(patterns)

    for pattern in patterns:
        coactivations += np.outer(pattern, pattern)

    coactivations /= number_of_patterns

    return coactivations


def create_orthogonal_canonical_representation(minicolumns, hypercolumns):
    aux = []
    for i in range(minicolumns):
        aux.append(i * np.ones(hypercolumns))

    return np.array(aux, dtype='int')


def build_network_representation(matrix, minicolumns, hypercolumns):
    network_representation = np.zeros((len(matrix), minicolumns * hypercolumns), dtype='int')

    for pattern, indexes in enumerate(matrix):
        for hypercolumn_index, minicolumn_index in enumerate(indexes):
            index = hypercolumn_index * minicolumns + minicolumn_index
            network_representation[pattern, index] = 1

    return network_representation


def get_weights_from_probabilities(pi, pj, pij, minicolumns, hypercolumns, small_number=10e-10):

    n_units = minicolumns * hypercolumns

    aux = np.copy(pi)
    aux[pi < small_number] = small_number
    beta = np.log10(aux)

    w = np.zeros((n_units, n_units))
    for index1, p1 in enumerate(pi):
        for index2, p2 in enumerate(pj):
            if p1 == 0 or p2 == 0:
                w[index1, index2] = 1
            elif pij[index1, index2] < small_number:
                w[index1, index2] = small_number
            else:
                w[index1, index2] = pij[index1, index2] / (p1 * p2)

    w = np.log10(w)

    return w, beta


def get_probabilities_from_network_representation(network_representation):
    n_patterns = network_representation.shape[0]
    n_units = network_representation.shape[1]

    pi = network_representation.sum(axis=0)

    pij = np.zeros((n_units, n_units))
    for pattern in network_representation:
        pij += pattern[:, np.newaxis] @ pattern[np.newaxis, :]

    pi = pi / n_patterns
    pij /= n_patterns

    return pi, pij

def get_w(P, p, diagonal_zero=True):

    outer = np.outer(p, p)

    w = log_epsilon(P) - log_epsilon(outer)
    if diagonal_zero:
        w[np.diag_indices_from(w)] = 0

    return w


def get_w_pre_post(P, p_pre, p_post, p=1.0, epsilon=1e-20, diagonal_zero=True):

    outer = np.outer(p_post, p_pre)

    # w = np.log(p * P) - np.log(outer)
    x = p * (P / outer)
    # w = np.log(x)
    w = log_epsilon(x, epsilon)

    if diagonal_zero:
        w[np.diag_indices_from(w)] = 0

    return w


def get_beta(p, epsilon=1e-10):

    probability = np.copy(p)
    probability[p < epsilon] = epsilon

    beta = np.log10(probability)

    return beta


def softmax(input_vector, G=1.0, minicolumns=2):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    w : list of numbers
    t : float

    Return
    ------
    a list of the same length as w of non-negative numbers

    Examples
    --------
    >>> softmax([0.1, 0.2])

    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])

    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])

    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    lower_bound = -600
    upper_bound = 600

    x = np.copy(input_vector)
    x_size = x.size
    x = np.reshape(x, (x_size // minicolumns, minicolumns))
    x = G * np.array(x)

    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound

    e = np.exp(x)
    dist = normalize_array(e)

    dist = np.reshape(dist, x_size)
    return dist


def normalize_array(array):

    return array / np.sum(array, axis=1)[:, np.newaxis]


def strict_max(x, minicolumns):
    """
    A strict max that returns an array with 1 where the maximum of every minicolumn is
    :param x: the array
    :param minicolumns: number of minicolumns
    :return: the stric_max of the array
    """

    x = np.reshape(x, (x.size // minicolumns, minicolumns))
    z = np.zeros_like(x)
    maxes = np.argmax(x, axis=1)
    for max_index, max_aux in enumerate(maxes):
        z[max_index, max_aux] = 1

    return z.reshape(x.size)


def normalize_p(p, hypercolumns, minicolumns):

    x = p.reshape((hypercolumns, minicolumns))
    x = x / np.sum(x, axis=1)[:, np.newaxis]

    return x.reshape(hypercolumns * minicolumns)


def simple_bcpnn_matrix(minicolumns, w_self, w_next, w_rest):

    w = np.ones((minicolumns, minicolumns)) * w_rest
    for i in range(minicolumns):
        w[i, i] = w_self

    for i in range(minicolumns -1):
        w[i + 1, i] = w_next

    return w


def fill_connection(w, state_from, state_to, minicolumns, value):
    for hypercolumn_from, minicolumn_from in enumerate(state_from):
        for hypercolum_to, minicolumn_to in enumerate(state_to):
            index_from = hypercolumn_from * minicolumns + minicolumn_from
            index_to = hypercolum_to * minicolumns + minicolumn_to
            w[index_to, index_from] = value


def fill_sequence(w, minicolumns, sequence, weights_values, extension, alpha):
    n_states = len(sequence)
    # For every state
    for state_index, value in enumerate(weights_values):
        state_from = sequence[state_index]
        effective_extension = min(extension + 1, n_states - state_index)
        # Fill everything under extension gets out of the bonds of the sequence
        for next_index in range(effective_extension):
            effective_value = value - next_index * alpha
            state_to = sequence[state_index + next_index]
            fill_connection(w, state_from, state_to, minicolumns, effective_value)

    # Fll the laste value
    last_state = sequence[-1]
    fill_connection(w, last_state, last_state, minicolumns, value)


def create_matrix_from_sequences_representation(minicolumns, hypercolumns, sequences, weights_collection,
                                                extension, alpha, w_min=None):
    # Create the matrix
    if w_min is None:
        w_min = min([min(x) for x in weights_collection]) - (extension + 1) * alpha
    w = np.ones((minicolumns * hypercolumns, minicolumns * hypercolumns)) * w_min
    # Fill it
    for sequence, weights_values in zip(sequences, weights_collection):
        fill_sequence(w, minicolumns, sequence, weights_values, extension, alpha)

    return w


def fill_connection_aditive(w, state_from, state_to, minicolumns, value, w_min):
    for hypercolumn_from, minicolumn_from in enumerate(state_from):
        for hypercolum_to, minicolumn_to in enumerate(state_to):
            index_from = hypercolumn_from * minicolumns + minicolumn_from
            index_to = hypercolum_to * minicolumns + minicolumn_to
            if value > w[index_to, index_from] :
                w[index_to, index_from] = value


def fill_sequence_aditive(w, minicolumns, sequence, weights_values, extension, alpha, w_min):
    n_states = len(sequence)
    # For every state
    for state_index, value in enumerate(weights_values):
        state_from = sequence[state_index]
        effective_extension = min(extension + 1, n_states - state_index)
        # Fill everything under extension gets out of the bonds of the sequence
        for next_index in range(effective_extension):
            effective_value = value - next_index * alpha
            state_to = sequence[state_index + next_index]
            fill_connection_aditive(w, state_from, state_to, minicolumns, effective_value, w_min)

    # Fll the laste value
    last_state = sequence[-1]
    fill_connection_aditive(w, last_state, last_state, minicolumns, value, w_min)


def create_matrix_from_sequences_representation_aditive(minicolumns, hypercolumns, sequences, weights_collection,
                                                extension, alpha, w_min=None):
    # Create the matrix
    if w_min is None:
        w_min = min([min(x) for x in weights_collection]) - (extension + 1) * alpha
    w = np.ones((minicolumns * hypercolumns, minicolumns * hypercolumns)) * w_min
    # Fill it
    for sequence, weights_values in zip(sequences, weights_collection):
        fill_sequence_aditive(w, minicolumns, sequence, weights_values, extension, alpha, w_min)

    return w


def create_weights_from_overlap_protocol(nn, dt, n_patterns, s, r, mixed_start, contiguous,
                                         training_time, inter_pulse_interval, inter_sequence_interval,
                                         epochs, resting_time, TimedInput):
    filtered = True
    minicolumns = nn.minicolumns
    hypercolumns = nn.hypercolumns

    tau_z_pre_ampa = nn.tau_z_pre_ampa
    tau_z_post_ampa = nn.tau_z_post_ampa

    seq1, seq2 = produce_overlaped_sequences(minicolumns, hypercolumns, n_patterns, s, r,
                                             mixed_start=mixed_start, contiguous=contiguous)

    nr1 = build_network_representation(seq1, minicolumns, hypercolumns)
    nr2 = build_network_representation(seq2, minicolumns, hypercolumns)

    # Get the first
    timed_input = TimedInput(nr1, dt, training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs,
                             resting_time=resting_time)

    S = timed_input.build_timed_input()
    z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
    z_post = timed_input.build_filtered_input_post(tau_z_post_ampa)

    pi1, pj1, P1 = timed_input.calculate_probabilities_from_time_signal(filtered=filtered)
    w_timed1 = get_weights_from_probabilities(pi1, pj1, P1, minicolumns, hypercolumns)
    t1 = timed_input.T_total

    # Get the second
    timed_input = TimedInput(nr2, dt, training_time, inter_pulse_interval=inter_pulse_interval,
                             inter_sequence_interval=inter_sequence_interval, epochs=epochs,
                             resting_time=resting_time)

    S = timed_input.build_timed_input()
    z_pre = timed_input.build_filtered_input_pre(tau_z_pre_ampa)
    z_post = timed_input.build_filtered_input_post(tau_z_post_ampa)
    t2 = timed_input.T_total

    pi2, pj2, P2 = timed_input.calculate_probabilities_from_time_signal(filtered=filtered)
    w_timed2 = get_weights_from_probabilities(pi2, pj2, P2, minicolumns, hypercolumns)
    t_total = t1 + t2

    # Mix
    pi_total = (t1 / t_total) * pi1 + ((t_total - t1) / t_total) * pi2
    pj_total = (t1 / t_total) * pj1 + ((t_total - t1) / t_total) * pj2
    P_total = (t1 / t_total) * P1 + ((t_total - t1) / t_total) * P2
    w_total, beta = get_weights_from_probabilities(pi_total, pj_total, P_total, minicolumns, hypercolumns)

    return seq1, seq2, nr1, nr2, w_total, beta


def produce_overlaped_sequences(minicolumns, hypercolumns, n_patterns, s, r, mixed_start=False, contiguous=True):
    n_r = int(r * n_patterns / 2)
    n_s = int(s * hypercolumns)
    n_size = int(n_patterns / 2)

    matrix = create_orthogonal_canonical_representation(minicolumns, hypercolumns)[:n_patterns]
    sequence1 = matrix[:n_size]
    sequence2 = matrix[n_size:]

    if mixed_start:
        start_index = 0
        end_index = n_r
    else:
        start_index = max(int(0.5 * (n_size - n_r)), 0)
        end_index = min(start_index + n_r, n_size)

    for index in range(start_index, end_index):
        if contiguous:
            sequence2[index, :n_s] = sequence1[index, :n_s]
        else:
            sequence2[index, ...] = sequence1[index, ...]
            sequence2[index, n_s:] = n_patterns + index

    if False:
        print(n_r)
        print(n_size)
        print(start_index)
        print(end_index)

    return sequence1, sequence2

