import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch.optim as optim
from numpyencoder import NumpyEncoder
from scipy import stats as scipy_stats

from ca_networks.ca_layers import *
from ca_networks.utils import convert_bundle_space_to_pt_data

sns.set_style('whitegrid')


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def compute_metrics(preds, targets):
    metrics = {}
    r = scipy_stats.linregress(preds, targets)[2]
    kendall_tau = scipy_stats.kendalltau(preds, targets).correlation

    metrics['r'] = r
    metrics['kendall_tau'] = kendall_tau
    mae = sklearn.metrics.mean_absolute_error(preds, targets)
    metrics['mae'] = mae
    return metrics


class Net(nn.Module):
    def __init__(self, input_dim: int, num_hidden_layers: int, num_units: int, layer_type: str, target_max: float,
                 ts: float = 1.0):
        super(Net, self).__init__()
        if layer_type == 'PlainNN':
            fc_layer = torch.nn.Linear
            self.activation_funcs = [torch.relu for _ in range(num_hidden_layers)]
            self.output_activation_function = torch.relu
        else:
            fc_layer = eval(layer_type)
            if type(ts) == float or type(ts) == int:
                self.ts = [ts] * (num_hidden_layers)
            else:
                self.ts = ts
            self.activation_funcs = [partial(ca_activation_func, t=t) for t in self.ts]
            self.output_activation_function = torch.nn.Identity()

        self._layer_type = layer_type
        self._num_hidden_layers = num_hidden_layers
        self._layer_type = layer_type
        self._target_max = target_max

        self.layers = []
        fc1 = fc_layer(input_dim, num_units)
        self.layers.append(fc1)
        for _ in range(num_hidden_layers - 1):
            self.layers.append(fc_layer(num_units, num_units))
        self.layers = torch.nn.ModuleList(self.layers)

        self.output_layer = fc_layer(num_units, 1) if layer_type == 'PlainNN' else fc_layer(num_units, 1, bias=False)
        self.output_activation_function = F.relu if layer_type == 'PlainNN' else torch.nn.Identity()
        self.dataset_info = None
        assert len(self.layers) == len(
            self.activation_funcs), 'Incorrect number of layers and activation functions.'

    def set_activation_functions(self, ts):
        assert len(self.layers) == len(ts), 'Incorrect number of layers and activation functions.'
        self.ts = ts
        self.activation_funcs = [partial(ca_activation_func, t=t) for t in ts]

    def forward(self, x):
        for layer, activation_func in zip(self.layers, self.activation_funcs):
            x = layer(x)
            x = activation_func(x)

        # Output layer
        x = self.output_activation_function(self.output_layer(x))
        return x

    def transform_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'transform_weights'):
                layer.transform_weights()
        if hasattr(self.output_layer, 'transform_weights'):
            self.output_layer.transform_weights()


def train(model, device, train_loader, optimizer, epoch, dataset_info, loss_func):
    total_loss = 0
    model.train()
    preds, targets = [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        preds.extend(output.detach().cpu().numpy().flatten().tolist())
        targets.extend(target.detach().cpu().numpy().flatten().tolist())
        loss = loss_func(output.flatten(), target.flatten())
        total_loss += float(loss) * len(preds)
        loss.backward()
        optimizer.step()
    metrics = {'loss': total_loss / len(train_loader)}
    preds, targets = (np.array(preds) * dataset_info['target_max']).tolist(), \
                     (np.array(targets) * dataset_info['target_max']).tolist()
    metrics.update(compute_metrics(preds, targets))
    return metrics


def test(model, device, loader, valid_true, epoch, dataset_info, loss_func, plot=False, log_path=None):
    model.eval()
    test_loss = 0
    correct = 0
    preds, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds.extend(output.detach().cpu().numpy().flatten().tolist())
            targets.extend(target.detach().cpu().numpy().flatten().tolist())
            test_loss += loss_func(output.flatten(), target.flatten(), reduction='sum').item()  # sum up batch loss

    preds, targets = (np.array(preds) * dataset_info['target_max']).tolist(), \
                     (np.array(targets) * dataset_info['target_max']).tolist()
    metrics = {}

    # Check 0-to-0 mapping
    metrics['0-to-0'] = int(np.all(model(torch.zeros((1, data.shape[1]))).detach().cpu().numpy() == 0))

    # Checking monotonicity
    inp = torch.zeros((1, data.shape[1]))
    outputs = []
    for i in range(data.shape[1]):
        inp[0, i] = 1
        outputs.append(float(model(inp).detach().cpu().numpy()))
    metrics['monotonicity_satisfied'] = int(sorted(outputs) == outputs)

    metrics.update(compute_metrics(preds, targets))
    eval_type = 'valid' if valid_true else 'test'

    if plot:
        dat_min, dat_max = min(min(preds), min(targets)), \
                           max(max(preds), max(targets))
        plt.figure(figsize=(4, 3))
        plt.scatter(np.array(targets), np.array(preds), s=1, alpha=0.01)
        plt.plot([dat_min, dat_max], [dat_min, dat_max], 'y')
        plt.ylabel('Pred')
        plt.xlabel('True')
        plt.title('Ep: {} | kt: {:.2f} | r: {:.2f} | mae: {:.4f}'.format(
            epoch, metrics['kendall_tau'], metrics['r'], metrics['mae']))
        plt.tight_layout()
        plt.show()
        plt.close()

    test_loss /= len(loader.dataset)
    metrics['loss'] = test_loss

    return metrics


def train_model(train_dataset, config, logs, val_dataset=None, test_dataset=None, log_path=None, eval_test=False,
                save_datasets=False):
    device = torch.device("cpu")

    model = Net(input_dim=config['input_dim'], layer_type=config['layer_type'],
                num_hidden_layers=config['num_hidden_layers'],
                num_units=config['num_units'], target_max=config['target_max'],
                ts=config['ts']).to(device)
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=.9, weight_decay=config['l2'])
    else:
        raise NotImplementedError()
    loss_func = eval(config['loss_func'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, num_workers=2)
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, num_workers=2)

    metrics = defaultdict(dict)
    last_train_loss = np.inf
    best_model = None
    best_epoch = 0

    reattempt = True
    attempts = 0
    while reattempt and attempts < 20:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config['epochs']))
        attempts += 1
        model.apply(weights_init)
        for epoch in range(1, config['epochs'] + 1):
            metrics['train'][epoch] = train(model, device, train_loader, optimizer, epoch, config,
                                            loss_func=loss_func)
            scheduler.step()

        if last_train_loss > metrics['train'][epoch]['loss']:
            best_model = pickle.loads(pickle.dumps(model))
            last_train_loss = metrics['train'][epoch]['loss']

        if np.isnan(metrics['train'][epoch]['kendall_tau']) or metrics['train'][epoch]['kendall_tau'] < 0 or \
                metrics['train'][epoch]['r'] < 0.9:
            reattempt = True
        else:
            reattempt = False

    model = best_model
    # Transform the weights
    model.transform_weights()
    if val_dataset is not None:
        metrics['val'][epoch] = test(model, device, val_loader, valid_true=True, plot=False, epoch=epoch,
                                     log_path=None, dataset_info=config, loss_func=loss_func)

    if eval_test:
        if test_dataset is not None:
            metrics['test'][epoch] = test(model, device, test_loader, valid_true=False, epoch=epoch, plot=True,
                                          log_path=None, dataset_info=config, loss_func=loss_func)
    logs['metrics'] = metrics
    if save_datasets:
        metrics['datasets'] = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset,
                               'target_max': config['target_max']}
        metrics['model'] = model
    if log_path is not None and not save_datasets:
        json.dump(logs, open(os.path.join(log_path, 'results.json'), 'w'))
    return model, logs


def get_training_data(SAT_instance, num_train_data, seed, bidder_id, layer_type, normalize, normalize_factor):
    train_dataset, val_dataset, test_dataset, dataset_info = convert_bundle_space_to_pt_data(
        'data/{}/{}_seed{}_all_bids.pkl'.format(
            SAT_instance.upper(), SAT_instance.upper(), seed), bidder_id,
        num_train_data=num_train_data, normalize=normalize, seed=seed, normalize_factor=normalize_factor)
    return train_dataset, val_dataset, test_dataset, dataset_info


def eval_config(seed, SAT_instance, num_train_data, bidder_id, layer_type, batch_size, num_hidden_layers,
                num_hidden_units, optimizer, epochs, loss_func, lr, l2, normalize, normalize_factor, eval_test=False,
                log_path=None, save_datasets=False):
    logs = defaultdict()
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, test_dataset, dataset_info = \
        get_training_data(SAT_instance=SAT_instance, num_train_data=num_train_data,
                          seed=seed, bidder_id=bidder_id, layer_type=layer_type, normalize=normalize,
                          normalize_factor=normalize_factor)

    config = {
        'batch_size': batch_size,
        'loss_func': loss_func,
        'epochs': epochs,
        'num_hidden_layers': num_hidden_layers,
        'num_units': num_hidden_units,
        'layer_type': layer_type,
        'input_dim': dataset_info['M'],
        'lr': lr,
        'target_max': dataset_info['target_max'],
        'optimizer': optimizer,
        'l2': l2,
        'ts': 1.0
    }

    model, logs = train_model(train_dataset, config, logs, val_dataset=val_dataset, test_dataset=test_dataset,
                              log_path=None, eval_test=eval_test, save_datasets=save_datasets)
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        json.dump(logs, open(os.path.join(log_path, '{}.json'.format(seed)), 'w'), indent=4, sort_keys=True,
                  separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)
    return model, logs


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-7, metavar='LR',
                        help='l2 reg.')
    parser.add_argument('--layer_type', type=str, default='CALayerAbs', metavar='layer type',
                        help='Layer type')
    parser.add_argument('--loss_func', type=str, default='F.l1_loss', metavar='loss function',
                        help='loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='optimizer',
                        help='Optimizer')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='Number of hidden layers in the MLP.')
    parser.add_argument('--num_units', type=int, default=100,
                        help='Number of units in hidden layers in the MLP.')
    parser.add_argument('--log-interval', type=int, default=10000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_train_data', type=int, default=100,
                        help='number of training data points to choose')
    parser.add_argument('--problem_instance', type=str, default='lsvm')
    parser.add_argument('--bidder_id', type=int, default=1, help='bidder id of the auction')
    parser.add_argument('--log_path', type=str, default=None, help='Where to save the results.')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='whether to normalize the target regression variables.')
    args = parser.parse_args()
    _, logs = \
        eval_config(seed=args.seed, SAT_instance=args.problem_instance, num_train_data=args.num_train_data,
                    bidder_id=args.bidder_id, layer_type=args.layer_type, batch_size=args.batch_size,
                    epochs=args.epochs, num_hidden_layers=args.num_hidden_layers, num_hidden_units=args.num_units,
                    optimizer=args.optimizer, eval_test=True, loss_func=args.loss_func, l2=args.l2, lr=args.lr,
                    normalize=args.normalize, normalize_factor=1.0, log_path=args.log_path)
    pass


if __name__ == '__main__':
    main()
