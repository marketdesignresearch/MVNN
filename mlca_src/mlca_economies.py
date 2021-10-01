import copy
import itertools
import logging
# Libs
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from functools import partial

import docplex
import numpy as np

import mlca_src.mlca_util as util
from mlca_src.mlca_nn_mip_torch import NN_MIP_TORCH as MLCA_NNMIP
from mlca_src.mlca_nn_pt import MLCA_NN
from mlca_src.mlca_wdp import MLCA_WDP

# from tensorflow.keras.backend import clear_session

# documentation
# http://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html





# %%

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def eval_bidder_nn(bidder, fitted_scaler, local_scaling_factor, NN_parameters, elicited_bids):
    bids = elicited_bids[bidder]
    logging.info(bidder)

    start = time.time()
    nn_model = MLCA_NN(X_train=bids[0], Y_train=bids[1], scaler=fitted_scaler,
                       local_scaling_factor=local_scaling_factor)  # instantiate class
    nn_model.initialize_model(model_parameters=NN_parameters[bidder])  # initialize model
    train_logs = nn_model.fit(epochs=NN_parameters[bidder]['epochs'],
                              batch_size=NN_parameters[bidder]['batch_size'],  # fit model to data
                              X_valid=None, Y_valid=None)
    end = time.time()
    logging.info('Time for ' + bidder + ': %s sec\n', round(end - start))
    return {bidder: [nn_model, train_logs]}


class MLCA_Economies:

    def __init__(self, SATS_auction_instance, SATS_auction_instance_seed, Qinit, Qmax, Qround, scaler,
                 local_scaling_factor):

        # STATIC ATTRIBUTES
        self.SATS_auction_instance = SATS_auction_instance  # auction instance from SATS: LSVM, GSVM or MRVM generated via PySats.py.
        self.SATS_auction_instance_allocation = None  # true efficient allocation of auction instance
        self.SATS_auction_instance_scw = None  # SATS_auction_instance.get_efficient_allocation()[1]  # social welfare of true efficient allocation of auction instance
        self.SATS_auction_instance_seed = SATS_auction_instance_seed  # auction instance seed from SATS
        self.bidder_ids = list(SATS_auction_instance.get_bidder_ids())  # bidder ids in this auction instance.
        self.bidder_names = list('Bidder_{}'.format(bidder_id) for bidder_id in self.bidder_ids)
        self.N = len(self.bidder_ids)  # number of bidders
        self.good_ids = set(SATS_auction_instance.get_good_ids())  # good ids in this auction instance
        self.M = len(self.good_ids)  # number of items
        self.Qinit = Qinit  # number of intial data points, i.e., bundle-value pairs (queried randomly per bidder prior to the elicitation procedure, different per bidder)
        self.Qmax = Qmax  # maximal number of possible value queries in the preference elicitation algorithm (PEA) per bidder
        self.Qround = Qround  # maximal number of marginal economies per auction round per bidder
        self.scaler = scaler  # scaler is a instance from sklearn.MinMaxScaler() used for scaling the values of the bundle-value pairs
        self.fitted_scaler = None  # fitted scaler to the initial bids
        self.mlca_allocation = None  # mlca allocation
        self.mlca_scw = None  # true social welfare of mlca allocation
        self.mlca_allocation_efficiency = None  # efficiency of mlca allocation
        self.MIP_parameters = None  # MIP parameters
        self.mlca_iteration = 0  # mlca iteration tracker
        self.revenue = 0  # sum of mlca payments
        self.relative_revenue = None  # relative revenue cp to SATS_auction_instance_scw
        self.number_of_optimization = {'normal': 0, 'bidder_specific': 0}
        subsets = list(map(list, itertools.combinations(self.bidder_ids,
                                                        self.N - 1)))  # orderedDict containing all economies and the corresponding bidder ids of the bidders which are active in these economies.
        subsets.sort(reverse=True)  # sort s.t. Marginal Economy (-0)-> Marginal Economy (-1) ->...
        self.economies = OrderedDict(list(('Marginal Economy -({})'.format(i), econ) for econ, i in zip(subsets, [
            [x for x in self.bidder_ids if x not in subset][0] for subset in subsets])))
        self.economies['Main Economy'] = self.bidder_ids
        self.economies_names = OrderedDict(list((key, ['Bidder_{}'.format(s) for s in value]) for key, value in
                                                self.economies.items()))  # orderedDict containing all economies and the corresponding bidder names (as strings) of the bidders which are active in these economies.
        self.efficiency_per_iteration = OrderedDict()  # storage for efficiency stat püer auction round
        self.efficient_allocation_per_iteration = OrderedDict()  # storage for efficent allocation per auction round
        self.mip_log = defaultdict(list)
        self.train_logs = defaultdict(dict)

        # DYNAMIC PER ECONOMY
        self.economy_status = OrderedDict(list((key, False) for key, value in
                                               self.economies.items()))  # boolean, status of economy: if already calculated.
        self.mlca_marginal_allocations = OrderedDict(list((key, None) for key, value in self.economies.items() if
                                                          key != 'Main Economy'))  # Allocation of the WDP based on the elicited bids
        self.mlca_marginal_scws = OrderedDict(list((key, None) for key, value in self.economies.items() if
                                                   key != 'Main Economy'))  # Social Welfare of the Allocation of the WDP based on the elicited bids
        self.elapsed_time_mip = OrderedDict(
            list((key, []) for key, value in self.economies.items()))  # stored MIP solving times per economy
        self.warm_start_sol = OrderedDict(list((key, None) for key, value in
                                               self.economies.items()))  # MIP SolveSolution object used as warm start per economy

        # DYNAMIC PER BIDDER
        self.mlca_payments = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # VCG-style payments in MLCA, calculated at the end
        self.mlca_payments_parts = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                                    self.bidder_ids))
        self.elicited_bids = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # R=(R_1,...,R_n) elicited bids per bidder
        self.current_query_profile = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                                      self.bidder_ids))  # S=(S_1,...,S_n) number of actual value queries, that takes into account if the same bundle was queried from a bidder in two different economies that it is not counted twice
        self.NN_parameters = OrderedDict(list(('Bidder_{}'.format(bidder_id), None) for bidder_id in
                                              self.bidder_ids))  # DNNs parameters as in the Class NN described.

        # DYNAMIC PER ECONOMY & BIDDER
        self.argmax_allocation = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in
            self.economies_names.items()))  # [a,a_restr] a: argmax bundles per bidder and economy, a_restr: restricted argmax bundles per bidder and economy
        self.NN_models = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in
            self.economies_names.items()))  # tf.keras NN models
        self.losses = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, []) for bidder_id in value))) for key, value in
            self.economies_names.items()))  # Storage for the MAE loss during training of the DNNs
        self.local_scaling_factor = local_scaling_factor

    def get_info(self, final_summary=False):
        if not final_summary: logging.warning('INFO')
        if final_summary: logging.warning('SUMMARY')
        logging.warning('-----------------------------------------------')
        logging.warning('Seed Auction Instance: %s', self.SATS_auction_instance_seed)
        logging.warning('Iteration of MLCA: %s', self.mlca_iteration)
        logging.warning('Number of Elicited Bids:')
        for k, v in self.elicited_bids.items():
            logging.warning(k + ': %s', v[0].shape[0] - 1)  # -1 because of added zero bundle
        logging.warning('Qinit: %s | Qround: %s | Qmax: %s', self.Qinit, self.Qround, self.Qmax)
        if (not final_summary) and (not len(self.efficiency_per_iteration.keys()) == 0): logging.warning(
            'Efficiency given elicited bids from iteration 0-%s: %s\n', self.mlca_iteration - 1,
            self.efficiency_per_iteration[self.mlca_iteration - 1])

    def get_number_of_elicited_bids(self, bidder=None):
        if bidder is None:
            return OrderedDict((bidder, self.elicited_bids[bidder][0].shape[0] - 1) for bidder in
                               self.bidder_names)  # -1, because null bundle was added
        else:
            return self.elicited_bids[bidder][0].shape[0] - 1

    def calculate_efficiency_per_iteration(self):
        logging.debug('')
        logging.debug('Calculate current efficiency:')
        allocation, objective = self.solve_WDP(self.elicited_bids, verbose=1)
        self.efficient_allocation_per_iteration[self.mlca_iteration] = allocation
        efficiency = self.calculate_efficiency_of_allocation(allocation=allocation,
                                                             allocation_scw=objective)  # -1 elicited bids after previous iteration
        logging.debug('Current efficiency: {}'.format(efficiency))
        self.efficiency_per_iteration[self.mlca_iteration] = efficiency
        return efficiency

    def set_NN_parameters(self, parameters):
        logging.debug('Set NN parameters')
        self.NN_parameters = OrderedDict(parameters)

    def set_MIP_parameters(self, parameters):
        logging.debug('Set MIP parameters')
        self.MIP_parameters = parameters

    def set_initial_bids(self, initial_bids=None, fitted_scaler=None, seed=None):
        logging.info('INITIALIZE BIDS')
        logging.info('-----------------------------------------------\n')
        if initial_bids is None:  # Uniform sampling (now with correct isLegacy Version)
            self.elicited_bids, self.fitted_scaler = util.initial_bids_mlca_unif(
                SATS_auction_instance=self.SATS_auction_instance,
                number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                scaler=self.scaler, seed=seed)
        else:
            logging.debug('Setting inputed initial bids of dimensions:')
            if not (list(initial_bids.keys()) == self.bidder_names):
                logging.info(
                    'Cannot set inputed initial bids-> sample uniformly.')  # Uniform sampling (now with correct isLegacy Version)
                self.elicited_bids, self.fitted_scaler = util.initial_bids_mlca_unif(
                    SATS_auction_instance=self.SATS_auction_instance,
                    number_initial_bids=self.Qinit, bidder_names=self.bidder_names,
                    scaler=self.scaler, seed=seed)
            else:
                for k, v in initial_bids.items():
                    logging.debug(k + ': X=%s, Y=%s', v[0].shape, v[1].shape)
                self.elicited_bids = initial_bids  # needed format: {Bidder_i:[bundles,values]}
                self.fitted_scaler = fitted_scaler  # fitted scaler to initial bids
                self.Qinit = [v[0].shape[0] for k, v in initial_bids.items()]

    def reset_argmax_allocations(self):
        self.argmax_allocation = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, [None, None]) for bidder_id in value))) for key, value in
            self.economies_names.items()))

    def reset_current_query_profile(self):
        self.current_query_profile = OrderedDict(
            list(('Bidder_{}'.format(bidder_id), None) for bidder_id in self.bidder_ids))

    def reset_NN_models(self):
        delattr(self, 'NN_models')
        self.NN_models = OrderedDict(list(
            (key, OrderedDict(list((bidder_id, None) for bidder_id in value))) for key, value in
            self.economies_names.items()))
        # clear_session()

    def reset_economy_status(self):
        self.economy_status = OrderedDict(list((key, False) for key, value in self.economies.items()))

    def solve_SATS_auction_instance(self):
        self.SATS_auction_instance_allocation, self.SATS_auction_instance_scw = self.SATS_auction_instance.get_efficient_allocation()

    def sample_marginal_economies(self, active_bidder, number_of_marginals):
        admissible_marginals = [x for x in list((self.economies.keys())) if
                                x not in ['Main Economy', 'Marginal Economy -({})'.format(active_bidder)]]
        return random.sample(admissible_marginals, k=number_of_marginals)

    def update_elicited_bids(self):
        for bidder in self.bidder_names:
            logging.info('UPDATE ELICITED BIDS: S -> R for %s', bidder)
            logging.info('---------------------------------------------')
            # update bundles
            self.elicited_bids[bidder][0] = np.append(self.elicited_bids[bidder][0], self.current_query_profile[bidder],
                                                      axis=0)
            # update values
            bidder_value_reports = self.value_queries(bidder_id=util.key_to_int(bidder),
                                                      bundles=self.current_query_profile[bidder])
            self.elicited_bids[bidder][1] = np.append(self.elicited_bids[bidder][1], bidder_value_reports,
                                                      axis=0)  # update values
            logging.info('CHECK Uniqueness of updated elicited bids:')
            check = len(np.unique(self.elicited_bids[bidder][0], axis=0)) == len(self.elicited_bids[bidder][0])
            logging.info('UNIQUE\n') if check else logging.debug('NOT UNIQUE\n')
        return (check)

    def update_current_query_profile(self, bidder, bundle_to_add):
        if bundle_to_add.shape != (self.M,):
            logging.debug('No valid bundle dim -> CANNOT ADD BUNDLE')
            return (False)
        if self.current_query_profile[bidder] is None:  # If empty can add bundle_to_add for sure
            logging.debug('Current query profile is empty -> ADD BUNDLE')
            self.current_query_profile[bidder] = bundle_to_add.reshape(1, -1)
            return (True)
        else:  # If not empty, first check for duplicates, then add
            if self.check_bundle_contained(bundle=bundle_to_add, bidder=bidder):
                return (False)
            else:
                self.current_query_profile[bidder] = np.append(self.current_query_profile[bidder],
                                                               bundle_to_add.reshape(1, -1), axis=0)
                logging.debug('ADD BUNDLE to current query profile')
                return (True)

    def value_queries(self, bidder_id, bundles):
        raw_values = np.array(
            [self.SATS_auction_instance.calculate_value(bidder_id, bundles[k, :]) for k in range(bundles.shape[0])])
        if self.fitted_scaler is None:
            logging.debug('Return raw value queries')
            return (raw_values)
        else:
            minI = int(round(self.fitted_scaler.data_min_[0] * self.fitted_scaler.scale_[0]))
            maxI = int(round(self.fitted_scaler.data_max_[0] * self.fitted_scaler.scale_[0]))
            logging.debug('*SCALING*')
            logging.debug('---------------------------------------------')
            logging.debug('raw values %s', raw_values)
            logging.debug('Return value queries scaled by: %s to the interval [%s,%s]',
                          round(self.fitted_scaler.scale_[0], 8), minI, maxI)
            logging.debug('scaled values %s', self.fitted_scaler.transform(raw_values.reshape(-1, 1)).flatten())
            logging.debug('---------------------------------------------')
            return (self.fitted_scaler.transform(raw_values.reshape(-1, 1)).flatten())

    def check_bundle_contained(self, bundle, bidder):
        if np.any(np.equal(self.elicited_bids[bidder][0], bundle).all(axis=1)):
            logging.info('Argmax bundle ALREADY ELICITED from {}\n'.format(bidder))
            return (True)
        if self.current_query_profile[bidder] is not None:
            if np.any(np.equal(self.current_query_profile[bidder], bundle).all(axis=1)):
                logging.info('Argmax bundle ALREADY QUERIED IN THIS AUCTION ROUND from {}\n'.format(bidder))
                return (True)
        return (False)

    def next_queries(self, economy_key, active_bidder):
        if not self.economy_status[economy_key]:  # check if economy already has been calculated prior
            self.estimation_step(economy_key=economy_key)
            self.optimization_step(economy_key=economy_key)

        if self.check_bundle_contained(bundle=self.argmax_allocation[economy_key][active_bidder][0],
                                       bidder=active_bidder):  # Check if argmax bundle already has been queried in R or S
            ########## FOR NULL BUNDLE EXPERIMENT
            # if self.check_bundle_contained(bundle=self.argmax_allocation[economy_key][active_bidder][0], bidder=active_bidder) or np.sum(self.argmax_allocation[economy_key][active_bidder][0]) == 0: # Check if argmax bundle already has been queried in R or S or equals null bundle
            ##########
            if self.current_query_profile[
                active_bidder] is not None:  # recalc optimization step with bidder specific constraints added
                Ri_union_Si = np.append(self.elicited_bids[active_bidder][0], self.current_query_profile[active_bidder],
                                        axis=0)
            else:
                Ri_union_Si = self.elicited_bids[active_bidder][0]
            ########### FOR NULL BUNDLE EXPERIMENT
            # null = np.zeros(self.M, dtype=int).reshape(1, -1)
            # if not np.any(np.equal(Ri_union_Si, null).all(axis=1)):
            # logging.info('NULL BUNDLE WILL BE ADDED TO BIDDER SPECIFIC CONSTRAINTS')
            # Ri_union_Si = np.append(Ri_union_Si, null, axis=0)
            ###########
            CTs = OrderedDict()
            CTs[active_bidder] = Ri_union_Si
            self.optimization_step(economy_key, bidder_specific_constraints=CTs)
            self.economy_status[economy_key] = True  # set status of economy to true
            allocation = copy.deepcopy(self.argmax_allocation[economy_key][active_bidder][1])
            assert not any((allocation == x).all() for x in Ri_union_Si), \
                'New allocation already elicited, but still returned. This is likely a CPLEX error, consider reducing integrality_tol in cplex.'
            self.number_of_optimization['bidder_specific'] += 1
            return allocation  # return constrained argmax bundle

        else:  # If argmax bundle has NOT already been queried
            self.economy_status[economy_key] = True  # set status of economy to true
            self.number_of_optimization['normal'] += 1
            return (self.argmax_allocation[economy_key][active_bidder][0])  # return regular argmax bundle

    def estimation_step(self, economy_key):
        logging.info('ESTIMATION STEP')
        logging.info('-----------------------------------------------')

        models = merge_dicts(*[
            partial(eval_bidder_nn, fitted_scaler=self.fitted_scaler, NN_parameters=self.NN_parameters,
                    elicited_bids=self.elicited_bids, local_scaling_factor=self.local_scaling_factor)(key) for key in
            self.economies_names[economy_key]])

        # Add train metrics
        trained_models = {}
        for bidder_id, (model, train_logs) in models.items():
            trained_models[bidder_id] = model
            if bidder_id not in self.train_logs[economy_key]:
                self.train_logs[economy_key][bidder_id] = []
            self.train_logs[economy_key][bidder_id].append(train_logs)
        '''
        if self.fitted_scaler is not None:
            logging.info('Rescaling nets to original bundle values:')
            for bidder, model in trained_models.items():
                bundle_space = np.unique(
                    np.array([np.random.choice([0, 1], self.M) for _ in range(100)], dtype=np.float32), axis=0)
                y_before_scaling = model.model(torch.from_numpy(bundle_space))
                s = float(model.model._target_max ** (1 / (model.model._num_hidden_layers + 1)))
                layer = 1
                for k, v in dict(model.model.named_parameters()).items():
                    weight_desc = k.split('.')
                    if weight_desc[1].isdigit():
                        layer = int(weight_desc[1]) + 1
                    elif weight_desc[0] == 'output_layer':
                        layer += 1
                    else:
                        raise ValueError('Unknown layer number')

                    if 'bias' in weight_desc:
                        v.data = v.data * s ** layer
                    elif 'weight' in weight_desc:
                        v.data = v.data * s
                    else:
                        raise ValueError('Unknown layer')

                # Adjust the t's in the activation functions
                model.model.set_activation_functions([s ** i for i in range(1, model.model._num_hidden_layers + 1)])
                y_after_scaling = model.model(torch.from_numpy(bundle_space))
                logging.info('{} - MAE: {:.8f}'.format(
                    bidder, F.l1_loss(y_after_scaling * float(self.fitted_scaler.scale_), y_before_scaling)))
        '''
        self.NN_models[economy_key] = trained_models

    def optimization_step(self, economy_key, bidder_specific_constraints=None):
        DNNs = OrderedDict(
            list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))

        if bidder_specific_constraints is None:
            logging.info('OPTIMIZATION STEP')
        else:
            logging.info('ADDITIONAL BIDDER SPECIFIC OPTIMIZATION STEP for {}'.format(
                list(bidder_specific_constraints.keys())[0]))
        logging.info('-----------------------------------------------')

        attempts = self.MIP_parameters['attempts_DNN_WDP']
        # NEW GSVM specific constraints
        if (self.SATS_auction_instance.get_model_name() == 'GSVM' and not self.SATS_auction_instance.isLegacy):
            GSVM_specific_constraints = True
            national_circle_complement = list(self.good_ids - set(self.SATS_auction_instance.get_goods_of_interest(6)))
            logging.info('########## ATTENTION ##########')
            logging.info('GSVM specific constraints: %s', GSVM_specific_constraints)
            logging.info('###############################\n')
        else:
            GSVM_specific_constraints = False
            national_circle_complement = None

        for attempt in range(1, attempts + 1):
            logging.debug('Initialize MIP')
            X = MLCA_NNMIP(DNNs, L=self.MIP_parameters['bigM'])
            if not self.MIP_parameters['mip_bounds_tightening']:
                X.initialize_mip(verbose=False,
                                 bidder_specific_constraints=bidder_specific_constraints,
                                 GSVM_specific_constraints=GSVM_specific_constraints,  # NEW
                                 national_circle_complement=national_circle_complement)  # NEW
            elif self.MIP_parameters['mip_bounds_tightening'] == 'IA':
                X.tighten_bounds_IA(upper_bound_input=[1] * self.M, lower_bound_input=[0] * self.M)
                X.initialize_mip(verbose=False,
                                 bidder_specific_constraints=bidder_specific_constraints,
                                 GSVM_specific_constraints=GSVM_specific_constraints,  # NEW
                                 national_circle_complement=national_circle_complement)  # NEW
            elif self.MIP_parameters['mip_bounds_tightening'] == 'LP':
                X.tighten_bounds_IA(upper_bound_input=[1] * self.M, lower_bound_input=[0] * self.M)
                X.tighten_bounds_LP(upper_bound_input=[1] * self.M, lower_bound_input=[0] * self.M)
                X.initialize_mip(verbose=False,
                                 bidder_specific_constraints=bidder_specific_constraints,
                                 GSVM_specific_constraints=GSVM_specific_constraints,  # NEW
                                 national_circle_complement=national_circle_complement)  # NEW
            try:
                logging.info('Solving MIP')
                logging.info('Attempt no: %s', attempt)
                if self.MIP_parameters['warm_start'] and self.warm_start_sol[economy_key] is not None:
                    logging.debug('Using warm start')
                    sol, log = X.solve_mip(
                        log_output=False,
                        time_limit=self.MIP_parameters['time_limit'],
                        mip_relative_gap=self.MIP_parameters['relative_gap'],
                        integrality_tol=self.MIP_parameters['integrality_tol'],
                        mip_start=docplex.mp.solution.SolveSolution(X.Mip, self.warm_start_sol[economy_key].as_dict()))
                    self.warm_start_sol[economy_key] = sol
                else:
                    sol, log = X.solve_mip(
                        log_output=False,
                        time_limit=self.MIP_parameters['time_limit'],
                        mip_relative_gap=self.MIP_parameters['relative_gap'],
                        integrality_tol=self.MIP_parameters['integrality_tol'])
                    self.warm_start_sol[economy_key] = sol

                for key, value in log.items():
                    self.mip_log[key].append(value)

                if bidder_specific_constraints is None:
                    logging.debug('SET ARGMAX ALLOCATION FOR ALL BIDDERS')
                    b = 0
                    for bidder in self.argmax_allocation[economy_key].keys():
                        self.argmax_allocation[economy_key][bidder][0] = X.x_star[b, :]
                        b = b + 1
                else:
                    logging.debug('SET ARGMAX ALLOCATION ONLY BIDDER SPECIFIC for {}'.format(
                        list(bidder_specific_constraints.keys())[0]))
                    for bidder in bidder_specific_constraints.keys():
                        b = X.get_bidder_key_position(
                            bidder_key=bidder)  # transform bidder_key into bidder position in MIP
                        if any((X.x_star[b, :] == x).all() for x in bidder_specific_constraints[bidder]):
                            logging.warning('Already elicited bundle returned.')
                            raise RuntimeError('Already elicited bundle returned.')
                        else:
                            self.argmax_allocation[economy_key][bidder][1] = X.x_star[b, :]  # now on position 1!

                for key, value in self.argmax_allocation[economy_key].items():
                    logging.debug(key + ':  %s | %s', value[0], value[1])

                self.elapsed_time_mip[economy_key].append(X.soltime)
                break
            except Exception:
                logging.warning('-----------------------------------------------')
                logging.warning('NOT SUCCESSFULLY SOLVED in attempt: %s \n', attempt)
                logging.warning(X.Mip.solve_details)
                if attempt == attempts:
                    X.Mip.export_as_lp(basename='UnsolvedMip_iter{}_{}'.format(self.mlca_iteration, economy_key),
                                       path=os.getcwd(), hide_user_names=False)
                    sys.exit('STOP, not solved succesfully in {} attempts\n'.format(attempt))
                # clear_session()
                logging.debug('REFITTING:')
                self.estimation_step(economy_key=economy_key)
            DNNs = OrderedDict(
                list((key, self.NN_models[economy_key][key].model) for key in list(self.NN_models[economy_key].keys())))
        del X
        del DNNs

    def calculate_mlca_allocation(self, economy='Main Economy'):
        logging.info('Calculate MLCA allocation: %s', economy)
        active_bidders = self.economies_names[economy]
        logging.debug('Active bidders: %s', active_bidders)
        allocation, objective = self.solve_WDP(
            elicited_bids=OrderedDict(list((k, self.elicited_bids.get(k, None)) for k in active_bidders)))
        logging.debug('MLCA allocation in %s:', economy)
        for key, value in allocation.items():
            logging.debug('%s %s', key, value)
        logging.debug('Social Welfare: %s', objective)
        # setting allocations
        if economy == 'Main Economy':
            self.mlca_allocation = allocation
            self.mlca_scw = objective
        if economy in self.mlca_marginal_allocations.keys():
            self.mlca_marginal_allocations[economy] = allocation
            self.mlca_marginal_scws[economy] = objective

    def solve_WDP(self, elicited_bids, verbose=0):  # rem.: objective always rescaled to true values
        bidder_names = list(elicited_bids.keys())
        if verbose == 1: logging.debug('Solving WDP based on elicited bids for bidder: %s', bidder_names)
        elicited_bundle_value_pairs = [np.concatenate((bids[0], np.asarray(bids[1]).reshape(-1, 1)), axis=1) for
                                       bidder, bids in
                                       elicited_bids.items()]  # transform self.elicited_bids into format for WDP class
        wdp = MLCA_WDP(elicited_bundle_value_pairs)
        wdp.initialize_mip(verbose=0)
        wdp.solve_mip(verbose)
        # TODO: check solution formater
        objective = wdp.Mip.objective_value
        allocation = util.format_solution_mip_new(Mip=wdp.Mip, elicited_bids=elicited_bundle_value_pairs,
                                                  bidder_names=bidder_names, fitted_scaler=self.fitted_scaler)
        if self.fitted_scaler is not None:
            if verbose == 1:
                logging.debug('')
                logging.debug('*SCALING*')
                logging.debug('---------------------------------------------')
                logging.debug('WDP objective scaled: %s:', objective)
                logging.debug('WDP objective value scaled by: 1/%s', round(self.fitted_scaler.scale_[0], 8))
            objective = float(self.fitted_scaler.inverse_transform([[objective]]))
            if verbose == 1:
                logging.debug('WDP objective orig: %s:', objective)
                logging.debug('---------------------------------------------')
        return (allocation, objective)

    def calculate_efficiency_of_allocation(self, allocation, allocation_scw, verbose=0):
        self.solve_SATS_auction_instance()
        efficiency = allocation_scw / self.SATS_auction_instance_scw
        if verbose == 1:
            logging.debug('Calculating efficiency of input allocation:')
            for key, value in allocation.items():
                logging.debug('%s %s', key, value)
            logging.debug('Social Welfare: %s', allocation_scw)
            logging.debug('Efficiency of allocation: %s', efficiency)
        return (efficiency)

    def calculate_vcg_payments(self, forced_recalc=False):
        logging.debug('Calculate payments')

        # (i) solve marginal MIPs
        for economy in list(self.economies_names.keys()):
            if not forced_recalc:
                if economy == 'Main Economy' and self.mlca_allocation is None:
                    self.calculate_mlca_allocation()
                elif economy in self.mlca_marginal_allocations.keys() and self.mlca_marginal_allocations[
                    economy] is None:
                    self.calculate_mlca_allocation(economy=economy)
                else:
                    logging.debug('Allocation for %s already calculated', economy)
            else:
                logging.debug('Forced recalculation of %s', economy)
                self.calculate_mlca_allocation(economy=economy)  # Recalc economy

        # (ii) calculate VCG terms for this economy
        for bidder in self.bidder_names:
            marginal_economy_bidder = 'Marginal Economy -({})'.format(util.key_to_int(bidder))
            p1 = self.mlca_marginal_scws[
                marginal_economy_bidder]  # social welfare of the allocation a^(-i) in this economy
            p2 = sum([self.mlca_allocation[i]['value'] for i in self.economies_names[
                marginal_economy_bidder]])  # social welfare of mlca allocation without bidder i
            self.mlca_payments[bidder] = round(p1 - p2, 2)
            self.mlca_payments_parts[bidder] = [p1, p2]
            logging.info('Payment %s: %s - %s  =  %s', bidder, p1, p2, self.mlca_payments[bidder])
        self.revenue = sum([self.mlca_payments[i] for i in self.bidder_names])
        self.relative_revenue = self.revenue / self.SATS_auction_instance_scw
        logging.info('Revenue: {} | {}% of SCW in efficient allocation\n'.format(self.revenue, self.relative_revenue))


# %%
print('Class MLCA_Economies imported')
