import argparse
import json
import random

import numpy as np
import torch

from ca_networks.main import eval_config

bidder_type_to_bidder_id = {
    'GSVM': {'regional': 0, 'national': 6},
    'LSVM': {'national': 0, 'regional': 1},
    'MRVM': {'national': 7, 'regional': 3, 'local': 0},
    'SRVM': {'national': 5, 'regional': 3, 'local': 0, 'high_frequency': 2}
}

network_type_to_layer_type = {
    'MVNN': 'CALayerReLUProjected',
    'NN': 'PlainNN'
}


def evaluate_network(cfg: dict, seed: int, SAT_instance: str, bidder_type: str, num_train_data: int, layer_type: str,
                     normalize: bool, normalize_factor: float, eval_test=False, save_datasets=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return eval_config(
        seed=seed, SAT_instance=SAT_instance, bidder_id=bidder_type_to_bidder_id[SAT_instance.upper()][bidder_type],
        layer_type=layer_type, batch_size=cfg['batch_size'], num_hidden_layers=cfg['num_hidden_layers'],
        num_hidden_units=int(max(1, np.round(cfg['num_neurons'] / cfg['num_hidden_layers']))), l2=cfg['l2'],
        lr=cfg['lr'], normalize_factor=normalize_factor, optimizer=cfg['optimizer'], num_train_data=num_train_data,
        eval_test=True, epochs=cfg['epochs'], loss_func=cfg['loss_func'], normalize=normalize, save_datasets=False)


def main():
    parser = argparse.ArgumentParser(description='Prediction Performance Evaluation')
    parser.add_argument('--domain', type=str, default='GSVM', help='SATS domain',
                        choices=['GSVM', 'LSVM', 'SRVM', 'MRVM'])
    parser.add_argument('--T', type=int, default=20, help='Num. training data')
    parser.add_argument('--bidder_type', type=str, default='national',
                        help='whether to normalize the target regression variables.')
    parser.add_argument('--seed', type=int, default=1, choices=[1], help='SATS auction instance seed.')
    parser.add_argument('--network_type', type=str, default='MVNN', choices=['MVNN', 'NN'],
                        help='Evaluate either UNN or NN.')
    args = parser.parse_args()
    hpo_results = json.load(open('prediction_performance_hpo_results.json', 'r'))

    config_dict = hpo_results[args.domain][str(args.T)][args.bidder_type][network_type_to_layer_type[args.network_type]]
    print('Selected winner hyperparameters', config_dict)
    model, logs = evaluate_network(
        config_dict, seed=args.seed, SAT_instance=args.domain, bidder_type=args.bidder_type,
        num_train_data=args.T, layer_type=network_type_to_layer_type[args.network_type],
        normalize=True if args.network_type == 'MVNN' else False,
        normalize_factor=1 if args.network_type == 'MVNN' else 500)
    train_logs = logs['metrics']['train'][config_dict['epochs']]
    val_logs = logs['metrics']['val'][config_dict['epochs']]
    test_logs = logs['metrics']['test'][config_dict['epochs']]

    print('Train metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(train_logs['r'], train_logs['kendall_tau']))
    print('Valid metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(val_logs['r'], val_logs['kendall_tau']))
    print('Test metrics \t| pearson corr.: {:.3f}, KT: {:.3f}'.format(test_logs['r'], test_logs['kendall_tau']))


if __name__ == '__main__':
    main()
