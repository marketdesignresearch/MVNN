# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:31:33 2020

@author: jakob
"""

"""
FILE DESCRIPTION:

This file implements the class NN (Neural Network) that has the following functionalities:
    0.CONSTRUCTOR:  __init__(self, X_train, Y_train, scaler)
        *(X_train,Y_train) is the training set of bundle-value pairs*.
        X_train = The bundles of items
        Y_train = The corresponding values for the bundles from a specific bidder. If alreday scaled to a vertai range than you also have to set the scaler variable in teh folowing.
        scaler =  A scaler instance from sklearn.preprocessing.*, e.g., sklearn.preprocessing.MinMaxScaler(), which was used to scale the Y_train variables before creating a NN instance.
                  This instance ins used in the class NN for rescaling errors to the original scale, i.e., it is used as scaler.inverse_transform().
    1. METHOD: initialize_model(self, model_parameters)
        model_parameters = the parameters specifying the neural network:
        This method initializes the attribute model in the class NN by defining the architecture and the parameters of the neural network.
    2. METHOD: fit(self, epochs, batch_size, X_valid=None, Y_valid=None, sample_weight=None)
        epochs = Number of epochs the neural network is trained
        batch_size = Batch size used in training
        X_valid = Test set of bundles
        Y_valid = Values for X_valid.
        sample_weight = weights vector for datapoints of bundle-value pairs.
        This method fits a neural network to data and returns loss numbers.
    3. METHOD: loss_info(self, batch_size, plot=True, scale=None)
        batch_size = Batch size used in training
        plot = boolean parameter if a plots for the goodness of fit should be executed.
        scale = either None or 'log' defining the scaling of the y-axis for the plots
        This method calculates losses on the training set and the test set (if specified) and plots a goodness of fit plot.

See example_nn.py for an example of how to use the class NN.

"""

import logging
from collections import defaultdict

import numpy as np
import torch

from ca_networks.main import train_model, test

__author__ = 'Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'


# %% NN Class for each bidder


class MLCA_NN:

    def __init__(self, X_train, Y_train, scaler, local_scaling_factor):
        self.M = X_train.shape[1]  # number of items
        self.X_train = X_train  # training set of bundles
        self.Y_train = Y_train  # bidder's values for the bundels in X_train
        self.X_valid = None  # test/validation set of bundles
        self.Y_valid = None  # bidder's values for the bundels in X_valid
        self.model_parameters = None  # neural network parameters
        self.model = None  # keras model, i.e., the neural network
        self.scaler = scaler  # the scaler used for initially scaling the Y_train values
        self.history = None  # return value of the model.fit() method from keras
        self.loss = None  # return value of the model.fit() method from keras
        self.local_scaling_factor = local_scaling_factor
        self.device = torch.device("cpu")

    def initialize_model(self, model_parameters):
        self.model_parameters = model_parameters

    def fit(self, epochs, batch_size, X_valid=None, Y_valid=None):
        # set test set if desired
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        target_max = self.Y_train.reshape(-1, 1).max() / self.local_scaling_factor \
            if self.local_scaling_factor is not None else 1.0

        self.Y_train = self.Y_train / target_max
        # fit model and validate on test set
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X_train.astype(np.float32)),
            torch.from_numpy(self.Y_train.reshape(-1).astype(np.float32)))

        config = {
            'batch_size': self.model_parameters['batch_size'],
            'epochs': self.model_parameters['epochs'],
            'num_hidden_layers': self.model_parameters['num_hidden_layers'],
            'num_units': self.model_parameters['num_hidden_units'],
            'layer_type': self.model_parameters['layer_type'],
            'input_dim': self.X_train.shape[1],
            'lr': self.model_parameters['lr'],
            'loss_func': self.model_parameters['loss_func'],
            'target_max': target_max,
            'optimizer': self.model_parameters['optimizer'],
            'l2': self.model_parameters['l2'],
            'ts': float(1)
        }
        logs = defaultdict()

        model, logs = train_model(train_dataset, config, logs)
        self.model = model

        self.history = logs
        logging.debug('loss: {:.7f}, kt: {:.4f}, r: {:.4f}'.format(
            logs['metrics']['train'][epochs]['loss'],
            logs['metrics']['train'][epochs]['kendall_tau'],
            logs['metrics']['train'][epochs]['r']))
        # get loss infos
        loss = logs['metrics']['train'][epochs]['loss']
        return logs['metrics']['train'][epochs]

    def evaluate(self, X, y):
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X).float(),
                torch.from_numpy(y.reshape(-1, 1)).float()),
            batch_size=16)
        metrics = test(self.model, self.device, dataloader, valid_true=False, plot=False, epoch=0, log_path=None,
                       dataset_info=self.dataset_info, loss_func=eval(self.model_parameters['loss_func']))
        return metrics['loss']

    def predict(self, X):
        self.model.eval()
        return self.model(torch.from_numpy(X)) * self.dataset_info['target_max']
