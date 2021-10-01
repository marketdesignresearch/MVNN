import argparse
import os
import pickle
from functools import partial
from multiprocessing import Pool

from sats.pysats import PySats
from util import generate_all_bundle_value_pairs

parser = argparse.ArgumentParser(description='Prediction Performance Evaluation')
parser.add_argument('--domain', type=str, default='GSVM', help='SATS domain',
                    choices=['GSVM', 'LSVM', 'SRVM', 'MRVM'])
parser.add_argument('--number_of_instances', type=int, default=1, help='Num. training data')

args = parser.parse_args()
path = 'data/' + args.domain + '/'
seeds_instances = list(range(1, 1 + args.number_of_instances))  # for instances in (1)


def eval_seed(model_name, seed):
    print(f'Generating all bundle value pairs in {model_name} for seed {seed}')

    # generate value model
    if model_name == 'GSVM':
        value_model = PySats.getInstance().create_gsvm(seed=seed)
    elif model_name == 'LSVM':
        value_model = PySats.getInstance().create_lsvm(seed=seed)
    elif model_name == 'SRVM':
        value_model = PySats.getInstance().create_srvm(seed=seed)
    elif model_name == 'MRVM':
        value_model = PySats.getInstance().create_mrvm(seed=seed)
    else:
        raise NotImplementedError('Unknown SATS model')

    # generate all bundle_value_pairs
    bundle_value_pairs = generate_all_bundle_value_pairs(value_model,
                                                         order=0 if model_name not in ['SRVM', 'MRVM'] else 2)

    # save
    filename_forsaving = f'{model_name}_seed{seed}_all_bids.pkl'
    os.makedirs(path, exist_ok=True)
    f = open(path + filename_forsaving, "wb")
    print('Saving results as:', filename_forsaving)
    pickle.dump(bundle_value_pairs, f)
    f.close()
    del bundle_value_pairs
    del value_model


for seed in seeds_instances:
    eval_seed(model_name=args.domain, seed=seed)