import pickle

import numpy as np
import torch


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


test_seeds_per_SATS_instance = {
    'LSVM': (101, 200),
    'GSVM': (101, 200),
    'SRVM': (101, 200),
    'MRVM': (101, 150)
}


def get_hyps_from_dirname(dirname):
    hyps = dirname.split('_')
    hyps_in_dir = {}
    try:
        _, _, _, _, _, _, _, problem_instance, layer_type, num_train_data, _, bidder_type, _, _ = hyps
    except ValueError as e:
        _, _, _, _, _, _, _, problem_instance, layer_type, num_train_data, _, bidder_type1, bidder_type2, _, _ = hyps
        bidder_type = bidder_type1 + '_' + bidder_type2
    bidder_type = bidder_type if bidder_type.lower() != 'high_frequency' else 'high_frequency'
    hyps_in_dir.update(
        {'problem_instance': problem_instance, 'layer_type': layer_type, 'num_train_data': int(num_train_data),
         'bidder_type': bidder_type})
    return hyps_in_dir


def convert_bundle_space_to_pt_data(path, bidder_id, normalize, normalize_factor, seed, num_train_data, val_ratio=0.2):
    """
    Return train, val and test dataset splits for a single bidder of the bundle space.
    """
    world = None
    if 'GSVM' in path:
        world = 'GSVM'
        N = 7
        M = 18
    elif 'LSVM' in path:
        world = 'LSVM'
        N = 6
        M = 18
    elif 'SRVM' in path:
        world = 'SRVM'
        N = 7
        M = 29
    elif 'MRVM' in path:
        world = 'MRVM'
        N = 10
        M = 98
    else:
        raise NotImplementedError('Unknown environment.')
    dataset_info = {'N': N, 'M': M, 'world': world}

    dataset = pickle.load(open(path, 'rb'))
    X, y = dataset[:, :M], dataset[:, M + bidder_id]
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X, y = unison_shuffled_copies(X, y)

    X_train, y_train = X[:num_train_data], \
                       y[:num_train_data]
    X_val, y_val = X[num_train_data:int(len(X) * 0.2 + num_train_data)], \
                   y[num_train_data:int(len(X) * 0.2 + num_train_data)]
    X_test, y_test = X[int(len(X) * 0.2 + num_train_data):], \
                     y[int(len(X) * 0.2 + num_train_data):]
    if normalize:
        y_train_max = max(y_train) * (1 / normalize_factor)
        dataset_info['target_max'] = y_train_max
        y_train, y_val, y_test = y_train / y_train_max, y_val / y_train_max, y_test / y_train_max
    else:
        dataset_info['target_max'] = 1.0

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                           torch.from_numpy(y_train.reshape(-1, 1)))
    val = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                         torch.from_numpy(y_val.reshape(-1, 1)))
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                          torch.from_numpy(y_test.reshape(-1, 1)))
    return train, val, test, dataset_info
