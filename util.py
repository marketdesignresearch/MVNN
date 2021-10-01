"""
FILE DESCRIPTION:

This file stores general helper functions.
"""

import itertools
import time
import random

# libs--
import numpy as np
from tqdm import tqdm

def timediff_d_h_m_s(td):
    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return -(td.days), -int(td.seconds / 3600), -int(td.seconds / 60) % 60, -(td.seconds % 60)
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


def generate_all_bundle_value_pairs(world, order=0, k=262144):
    N = world.get_bidder_ids()
    M = world.get_good_ids()
    print()
    if order == 0:
        bundle_space = list(itertools.product([0, 1], repeat=len(M)))
    elif order == 1:
        bundle_space = [[int(b) for b in bin(2 ** len(M) + k)[3:][::-1]] for k in np.arange(2 ** len(M))]
    elif order == 2:
        # for mrvm and srvm space too large -> sample instead
        bundle_space = [np.random.choice([0, 1], len(M)) for _ in range(k)]
        # Only use unique samples.
        bundle_space = np.unique(np.array(bundle_space), axis=0)
    else:
        raise NotImplementedError('Order must be either 0 or 1')
    s = time.time()
    bundle_value_pairs = np.array(
        [list(x) + [world.calculate_value(bidder_id, x) for bidder_id in N] for x in tqdm(bundle_space)])
    e = time.time()
    print('Elapsed sec: ', round(e - s))
    return (bundle_value_pairs)
