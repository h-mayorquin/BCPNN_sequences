import numpy as np


def get_weights(manager, from_pattern, to_pattern, mean=True):

    w_self = manager.nn.w_ampa[from_pattern, from_pattern]
    w_next = manager.nn.w_ampa[to_pattern, from_pattern]
    if mean:
        w_rest = np.mean(manager.nn.w_ampa[(to_pattern + 1):, from_pattern])
    else:
        w_rest = np.max(manager.nn.w_ampa[(to_pattern + 1):, from_pattern])
    return w_self, w_next, w_rest

def create_artificial_matrix(hypercolumns, minicolumns, number_of_patterns, value, inhibition, decay_factor,
                             extension, diagonal_zero, diagonal_across, diagonal_value, sequence_decay=1.0,
                             free_attractor=False, free_attractor_value=0.5):
    # Creat the small matrix
    w_small = np.ones((minicolumns, minicolumns)) * inhibition
    for i in range(number_of_patterns):
        if i < number_of_patterns - extension - 1:
            aux = extension
            for j in range(aux):
                w_small[i + 1 + j, i] = value * (decay_factor ** j) * (sequence_decay ** i)
        else:
            aux = number_of_patterns - i - 1
            for j in range(aux):
                w_small[i + 1 + j, i] = value * (decay_factor ** j) * (sequence_decay ** i)

    # Add identity
    if diagonal_across:
        w_small[:number_of_patterns, :number_of_patterns] += np.diag(np.ones(number_of_patterns) * (diagonal_value - inhibition))

    # Add free attractor
    if free_attractor:
        w_small[number_of_patterns:, number_of_patterns:] = free_attractor_value

    # Create the big matrix
    w = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    for j in range(hypercolumns):
        for i in range(hypercolumns):
            w[i * minicolumns:(i + 1) * minicolumns, j * minicolumns:(j + 1) * minicolumns] = w_small

            # Remove diagonal
        if diagonal_zero:
            w[np.diag_indices_from(w)] = 0

    return w


def calculate_distance_from_history(history, patterns, normalize=True):

    o = history['o']
    distances = np.zeros((o.shape[0], len(patterns)))

    for index, state in enumerate(o):
        diff = state - patterns
        dis = np.linalg.norm(diff, ord=1, axis=1)
        distances[index] = dis

    if normalize:
        distances = distances / np.sum(distances, axis=1)[:, np.newaxis]
    return distances


def calculate_angle_from_history(manager):
    """
    :param manager: A manager of neural networks, it is used to obtain the history of the activity and
     the patterns that were stored

    :return: A vector with the distances to the stored patterns. This vector will be as long as the number of points
     in time times the number of pattern stores
    """
    history = manager.history
    patterns_dic = manager.patterns_dic
    stored_pattern_indexes = manager.stored_patterns_indexes

    o = history['o'][1:]
    if o.shape[0] == 0:
        raise ValueError('You did not record the history of unit activities o')

    distances = np.zeros((o.shape[0], manager.nn.minicolumns))

    for index, state in enumerate(o):
        # Obtain the dot product between the state of the network at each point in time and each pattern
        nominator = [np.dot(state, patterns_dic[pattern_index]) for pattern_index in stored_pattern_indexes]
        # Obtain the norm of both the state and the patterns to normalize
        denominator = [np.linalg.norm(state) * np.linalg.norm(patterns_dic[pattern_index])
                       for pattern_index in stored_pattern_indexes]

        # Get the angles and store them
        dis = [a / b for (a, b) in zip(nominator, denominator)]
        distances[index, stored_pattern_indexes] = dis

    return distances


def calculate_winning_pattern_from_distances(distances):
    # Returns the number of the winning pattern
    return np.argmax(distances, axis=1)


def calculate_patterns_timings(winning_patterns, dt, remove=0):
    """

    :param winning_patterns: A vector with the winning pattern for each point in time
    :param dt: the amount that the time moves at each step
    :param remove: only add the patterns if they are bigger than this number, used a small number to remove fluctuations

    :return: pattern_timins, a vector with information about the winning pattern, how long the network stayed at that
     configuration, when it got there, etc
    """

    # First we calculate where the change of pattern occurs
    change = np.diff(winning_patterns)
    indexes = np.where(change != 0)[0]

    # Add the end of the sequence
    indexes = np.append(indexes, winning_patterns.size - 1)

    patterns = winning_patterns[indexes]
    patterns_timings = []

    previous = 0
    for pattern, index in zip(patterns, indexes):
        time = (index - previous + 1) * dt  # The one is because of the shift with np.change
        if time >= remove:
            patterns_timings.append((pattern, time, previous*dt, index * dt))
        previous = index

    return patterns_timings


def calculate_timings(manager, remove=0.005):

    angles = calculate_angle_from_history(manager)
    winning_patterns = calculate_winning_pattern_from_distances(angles)
    timings = calculate_patterns_timings(winning_patterns, manager.dt, remove=remove)

    return timings


def calculate_recall_time_quantities(manager, T_recall, T_cue, n, sequences):

    # If the manager does not have the patterns yet, get them from the sequences
    if not manager.stored_patterns_indexes:  # This test for empty list
        aux = [element for sequence in sequences for element in sequence]
        manager.stored_patterns_indexes = list(set(aux))
        manager.n_patterns = len(manager.stored_patterns_indexes)

    # Calculate the timings
    success = calculate_recall_success_sequences(manager, T_recall, T_cue, n, sequences)[0]
    timings = calculate_timings(manager, remove=0.010)
    patterns = [x[0] for x in timings]

    # Check correct subsequence recalling
    flag = subsequence(patterns, sequences[0])
    n_min = min(len(sequences[0]), len(timings))

    if flag:
        time = [x[1] for x in timings[:n_min]]
        total_sequence_time = sum(time)
        mean = np.mean(time[1:-1])
        std = np.std(time[1:-1])
    else:
        total_sequence_time = 0
        mean = 0
        std = 0

    return total_sequence_time, mean, std, success, timings


def calculate_compression_factor(manager, training_time, exclude_extrema=True, remove=0):
    """
    Calculate compression factors for the timings

    :param manager: a Network manager object
    :param training_time:  the time it took fo train each pattern
    :param exclude_extrema: exluce the beggining and the end of the recall (the first one is the cue, the last one
    takes a while to die)
    :param remove: only take into account states that last longer than the remove value

    :return: compression value, a list with the compression values for each list
    """

    if exclude_extrema:
        indexes = manager.stored_patterns_indexes[1:-1]
    else:
        indexes = manager.stored_patterns_indexes

    timings = calculate_timings(manager, remove=remove)

    compression = [training_time / timings[pattern_index][1] for pattern_index in indexes]

    return compression


def calculate_recall_success(manager, T_recall,  I_cue, T_cue, n, patterns_indexes):

    n_patterns = len(patterns_indexes)
    successes = 0
    for i in range(n):
        manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue)

        distances = calculate_angle_from_history(manager)
        winning = calculate_winning_pattern_from_distances(distances)
        timings = calculate_patterns_timings(winning, manager.dt, remove=0.010)
        pattern_sequence = [x[0] for x in timings]

        if pattern_sequence[:n_patterns] == patterns_indexes:
            successes += 1

    success_rate = successes * 100.0/ n

    return success_rate


def calculate_recall_success_sequences(manager, T_recall, T_cue, n, sequences):
    successes = []
    total_sequences = len(sequences)
    for n_recall in range(total_sequences):
        sequence_to_recall = sequences[n_recall]
        I_cue = sequence_to_recall[0]

        success = calculate_recall_success(manager, T_recall, I_cue, T_cue, n, patterns_indexes=sequence_to_recall)
        successes.append(success)

    return successes


# Functions to extract connectivity
def calculate_total_connections(manager, from_pattern, to_pattern, ampa=False, normalize=True):

    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    hypercolumns = manager.nn.hypercolumns
    minicolumns = manager.nn.minicolumns

    from_pattern_j = from_pattern
    to_pattern_i = to_pattern

    weights = 0.0
    pattern_i_indexes = [int(to_pattern_i + i * minicolumns) for i in range(hypercolumns)]
    pattern_j_indexes = [int(from_pattern_j + j * minicolumns) for j in range(hypercolumns)]

    for j_index in pattern_j_indexes:
        weights += w[pattern_i_indexes, j_index].sum()

    norm = (hypercolumns * hypercolumns)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_last_pattern_to_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    final_pattern = n_patterns - 1
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[free_attractor_indexes, final_pattern].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_free_attractor_to_first_pattern(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    first_pattern = 0
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[first_pattern, free_attractor_indexes].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_first_pattern_to_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    first_pattern = 0
    free_attractor_indexes = np.arange(n_patterns, minicolumns, dtype='int')
    weights = w[free_attractor_indexes, first_pattern].sum()

    norm = len(free_attractor_indexes)
    if normalize:
        weights /= norm

    return weights


def calculate_connections_among_free_attractor(manager, ampa=False, normalize=True):
    if ampa:
        w = manager.nn.w_ampa
    else:
        w = manager.nn.w

    n_patterns = manager.n_patterns
    minicolumns = manager.nn.minicolumns

    weights = w[n_patterns:minicolumns, n_patterns:minicolumns].sum()

    norm = (minicolumns - n_patterns) ** 2
    if normalize:
        weights /= norm

    return weights


def get_excitation(index, w):
    total_connectivity_weights = w[index, :]
    exc_indexes = total_connectivity_weights > 0
    excitation = total_connectivity_weights[exc_indexes]

    return excitation


def get_inhibition(index, w):
    total_connectivity_weights = w[index, :]
    inh_indexes = total_connectivity_weights < 0
    inhibition = total_connectivity_weights[inh_indexes]

    return inhibition


def calculate_excitation_inhibition_ratio(nn, sequences, ampa=False):
    """
    Calculates the average ratio of excitatory to inhibitory weight on the network
    :param nn: the neural network
    :param sequences: the sequence of indexes
    :param ampa: wehther you want the results for ampa or nmda
    :return: mean, var and list of ratios
    """
    if ampa:
        w_use = nn.w_ampa
    else:
        w_use = nn.w

    w = np.copy(w_use)

    total_exc = []
    total_inh = []

    for index in sequences[0]:
        excitation = get_excitation(index, w)
        inhibition = get_inhibition(index, w)

        total_exc.append(np.sum(excitation))
        total_inh.append(np.sum(inhibition))

    ratios = [x / -y for (x, y) in zip(total_exc, total_inh)]

    return np.mean(ratios), np.var(ratios), ratios


def calculate_excitation(nn, sequences, ampa=False):
    if ampa:
        w_use = nn.w_ampa
    else:
        w_use = nn.w

    w = np.copy(w_use)

    total_exc = []

    for index in sequences[0]:
        excitation = get_excitation(index, w)
        total_exc.append(np.sum(excitation))

    return np.mean(total_exc), np.var(total_exc), total_exc


def calculate_inhibition(nn, sequences, ampa=False):
    if ampa:
        w_use = nn.w_ampa
    else:
        w_use = nn.w

    w = np.copy(w_use)

    total_inh = []

    for index in sequences[0]:
        inhibition = get_inhibition(index, w)
        total_inh.append(np.sum(inhibition))

    return np.mean(total_inh), np.var(total_inh), total_inh


def subsequence(sub, sequence):
    """
    Calculates whether sub is a sub-sequence of sequence.
    Returns tree if it is the case

    :param sub: the sub-sequence
    :param sequence: the sequence
    :return: bool, true if it is indeed a sub-sequence
    """
    flag = True
    n_sub =len(sub)
    n_sequence = len(sequence)
    index = 0
    while index < n_sub and index < n_sequence:
        if sub[index] != sequence[index]:
            flag = False
            break
        index += 1

    return flag
