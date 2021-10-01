# Libs
import logging
from collections import OrderedDict

import docplex.mp.model as cpx
import numpy as np
import pandas as pd

# %% Neural Net Optimization Class


class NN_MIP_TORCH:
    '''
    This implements the class NN_MIP_TORCH.
    This class is used for solving MIP reformulations of the neural network-based Winner Determination Problems
        (i) for plain NNs as described in https://arxiv.org/abs/1907.05771 (Section 4) and
        (ii) for monotone-value-neural-networks (UNNs) as described in ADDLINK.
    The NNs are implemented via PYTORCH.
    '''

    def __init__(self,
                 models,
                 L=None):

        self.M = list(models[list(models.keys())[0]].parameters())[0].shape[
            1]  # number of items in the value model = dimension of input layer
        self.Models = models  # dict of pytorch models
        # type of NNs (currently all NNs must be of the same type)
        if 'ca' in list(self.Models.values())[0]._layer_type.lower():
            self.NN_type = 'UNN'
        else:
            self.NN_type = 'PlainNN'
        self.sorted_bidders = list(self.Models.keys())  # sorted list of bidders
        self.sorted_bidders.sort()
        self.N = len(models)  # number of bidders
        self.Mip = cpx.Model(name="NeuralNetworksMixedIntegerProgram")  # docplex instance

        self.z = {}  # MIP variable: see paper
        self.eta = {}  # MIP variable: see paper
        self.y = {}  # MIP variable: see paper
        self.mu = {}  # MIP variable: see paper
        self.s = {}  # MIP variable: see paper

        self.x_star = np.ones(shape=(self.N, self.M), dtype=int) * (-1)  # optimal allocation (-1=not yet solved)
        self.L = L  # global big-M variable: see paper
        self.soltime = None  # timing

        self.z_help = {}  # helper variable for bound tightening
        self.eta_help = {}  # helper variable for bound tightening
        self.y_help = {}  # helper variable for bound tightening
        self.mu_help = {}  # helper variable for bound tightening

        self.removed_activation_cts = 0  # counts the removed UNN or ReLU cts

        if self.NN_type == 'UNN':
            # IA bounds for z variable (preactivated!!!!)
            self.upper_bounds_z = OrderedDict(
                list((bidder_name, {idx + 1: val for
                                    idx, val in enumerate(
                        np.array([None] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                        in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            self.lower_bounds_z = OrderedDict(
                list((bidder_name, {idx + 1: val for
                                    idx, val in
                                    enumerate(
                                        np.array([None] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                                        in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            # Boolean variable if constraints for UNN-specific activation function are active
            self.active = OrderedDict(
                list((bidder_name, {idx + 1: val for
                                    idx, val in enumerate(
                        np.array([True] * layer.shape[0]).reshape(-1, 1) for layer
                        in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            # big-M variables
            self.L1 = OrderedDict(
                list((bidder_name,
                      {idx + 1: val for idx, val in
                       enumerate(np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                                 in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            self.L2 = OrderedDict(
                list((bidder_name,
                      {idx + 1: val for idx, val in
                       enumerate(np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                                 in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            self.L3 = OrderedDict(
                list((bidder_name,
                      {idx + 1: val for idx, val in
                       enumerate(np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                                 in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
            self.L4 = OrderedDict(
                list((bidder_name,
                      {idx + 1: val for idx, val in
                       enumerate(np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                                 in self._get_model_layers(bidder_name, exc_layer_type=['output']))}) for
                     bidder_name in self.sorted_bidders))
        else:
            # IA bounds for z variable (preactivated!!!!)
            self.upper_bounds_z = OrderedDict(
                list((bidder_name, {idx + 1: val for
                                    idx, val in enumerate(
                        np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                        in self._get_model_layers(bidder_name))}) for
                     bidder_name in self.sorted_bidders))
            self.upper_bounds_s = OrderedDict(
                list((bidder_name, {idx + 1: val for
                                    idx, val in enumerate(
                        np.array([self.L] * layer.shape[0], dtype=np.float32).reshape(-1, 1) for layer
                        in self._get_model_layers(bidder_name))}) for
                     bidder_name in self.sorted_bidders))

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M + 1)]
        D.loc['Sum'] = D.sum(axis=0)
        print(D)

    def solve_mip(self,
                  log_output=False,
                  time_limit=None,
                  mip_relative_gap=None,
                  integrality_tol=None,
                  mip_start=None):
        # add a warm start
        if mip_start is not None:
            # self.Mip # not sure why this is there JW?
            self.Mip.add_mip_start(mip_start)
        # set time limit
        if time_limit is not None:
            self.Mip.set_time_limit(time_limit)
        # set mip relative gap
        if mip_relative_gap is not None:
            self.Mip.parameters.mip.tolerances.mipgap = mip_relative_gap
        # set mip integrality tolerance
        if integrality_tol is not None:
            self.Mip.parameters.mip.tolerances.integrality.set(integrality_tol)

        logging.info('')
        logging.info('Solve MIP')
        logging.info('-----------------------------------------------')
        # self.Mip.parameters.mip.cuts.flowcovers = 1
        # self.Mip.parameters.mip.cuts.mircuts = 1
        logging.info(f'MIP warm-start {bool(mip_start)}')
        logging.info('MIP time Limit of %s', self.Mip.get_time_limit())
        logging.info('MIP relative gap %s', self.Mip.parameters.mip.tolerances.mipgap.get())
        logging.info('MIP integrality tol %s', self.Mip.parameters.mip.tolerances.integrality.get())

        # solve MIP
        Sol = self.Mip.solve(log_output=log_output)
        # get solution details
        try:
            self.soltime = Sol.solve_details._time
        except Exception:
            self.soltime = None
        mip_log = self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for j in range(0, self.M):
                self.x_star[i, j] = int(self.z[(i, 0, j)].solution_value)
        return Sol, mip_log

    def log_solve_details(self,
                          solved_mip):
        details = solved_mip.get_solve_details()
        logging.info('\nSolve Details')
        logging.info('-----------------------------------------------')
        logging.info(f'NN-Type : {self.NN_type}')
        if self.NN_type == 'UNN':
            logging.info(f' - number of UNN-activation-function-constraints removed: {self.removed_activation_cts}')
        else:
            logging.info(f' - number of ReLU-constraints removed: {self.removed_activation_cts}')
        logging.info('Status  : %s', details.status)
        logging.info('Time    : %s sec', round(details.time))
        logging.info('Problem : %s', details.problem_type)
        logging.info('Rel. Gap: {} %'.format(details.mip_relative_gap))
        logging.debug('N. Iter : %s', details.nb_iterations)
        logging.debug('Hit Lim.: %s', details.has_hit_limit())
        logging.debug('Objective Value: %s', solved_mip.objective_value)
        logging.info('\n')
        return {'n_iter': details.nb_iterations, 'rel. gap': details.mip_relative_gap,
                'hit_limit': details.has_hit_limit(), 'time': details.time}

    def summary(self):
        print('\n################################ SUMMARY ################################')
        print(f'NN Type: {self.NN_type}\n')
        print('----------------------------- OBJECTIVE --------------------------------')
        print(self.Mip.get_objective_expr(), '\n')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Objective Value: Not yet solved!\n")
        print('----------------------------- SOLVE STATUS -----------------------------')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics())
        if self.NN_type == 'UNN':
            print(f' - number of UNN-activation-function-constraints removed: {self.removed_activation_cts}\n')
        else:
            print(f' - number of ReLu-constraints removed: {self.removed_activation_cts}\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('----------------------------- OPT ALLOCATION ----------------------------')
        self.print_optimal_allocation()
        print('#########################################################################')
        print('\n')
        return (' ')

    def print_mip_constraints(self):
        print('\n############################### CONSTRAINTS ###############################')
        k = 0
        for m in range(0, self.Mip.number_of_constraints):
            if self.Mip.get_constraint_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_constraint_by_index(m))
                k = k + 1
        print('#########################################################################')
        print('\n')

    def _get_model_weights(self,
                           key,
                           layer_type=['layers']):
        model_weights = []
        for k, v in dict(self.Models[key].named_parameters()).items():
            weight_desc = k.split('.')
            if any(weight_desc[0] in layer for layer in layer_type):
                if not ('bias' in k or 'weight' in k):
                    raise NotImplementedError('Unknown weight type encountered: {}'.format(k))
                else:
                    model_weights.append(v.detach().cpu().numpy())
        return model_weights

    def _get_model_layers(self,
                          key,
                          exc_layer_type=[]):
        layers = []
        for k, v in dict(self.Models[key].named_parameters()).items():
            if 'bias' not in k:
                if all(exc_type not in k for exc_type in exc_layer_type):
                    layers.append(v)
        return layers

    def _clean_weights(self,
                       Wb):
        for v in range(0, len(Wb) - 2, 2):
            Wb[v][abs(Wb[v]) <= 1e-8] = 0
            Wb[v + 1][abs(Wb[v + 1]) <= 1e-8] = 0
            zero_rows = np.where(np.logical_and((Wb[v] == 0).all(axis=0), Wb[v + 1] == 0))[0]
            if len(zero_rows) > 0:
                logging.debug('Clean Weights (rows) %s', zero_rows)
                Wb[v] = np.delete(Wb[v], zero_rows, axis=1)
                Wb[v + 1] = np.delete(Wb[v + 1], zero_rows)
                Wb[v + 2] = np.delete(Wb[v + 2], zero_rows, axis=0)
        return (Wb)

    def _add_matrix_constraints(self,
                                i,
                                verbose=False):
        layer = 1
        key = self.sorted_bidders[i]
        if verbose is True:
            logging.debug('\nAdd Matrix constraints: %s', key)
        ##############
        # 1) UNN MIP #
        ##############
        if self.NN_type == 'UNN':
            ts = self.Models[key].ts

            # Wb = self._clean_weights(self._get_model_weights(key)) with weights cleaning
            Wb = self._get_model_weights(key)
            assert len(Wb) % 2 == 0, 'Currently weights and biases should always be coupled.'

            for v in range(0, len(Wb), 2):  # loop over layers
                if verbose is True:
                    logging.debug('\nLayer: %s', layer)
                W = Wb[v]
                if verbose is True:
                    logging.debug('W: %s', W.shape)
                b = Wb[v + 1]
                # Negate the bias like in the layer
                if verbose is True:
                    logging.debug('b: %s', b.shape)
                R, J = W.shape
                # decision variables
                if v == 0:
                    self.z.update(
                        {(i, 0, j): self.Mip.binary_var(name="x({})_{}".format(i, j)) for j in
                         range(0, J)})  # binary variables for allocation
                self.z.update({(i, layer, r): self.Mip.continuous_var(name="z({},{})_{}".format(i, layer, r)) for r in
                               range(0, R)})  # output value variables after activation

                # check if variables are needed or if we can remove them
                for r in range(0, R):
                    # Case 1 "for sure in the zero range of the UNN-activation function" (given IA Bounds on Neurons)
                    if (self.upper_bounds_z[key][layer][r][0] <= 0):
                        self.active[key][layer][r][0] = False
                    # Case 2: "for sure in the linear range of the UNN-activation function" (given IA Bounds on Neurons)
                    elif (self.upper_bounds_z[key][layer][r][0] > 0 and self.upper_bounds_z[key][layer][r][0] <= ts[
                        layer - 1] and self.lower_bounds_z[key][layer][r][0] < ts[layer - 1] and
                          self.lower_bounds_z[key][layer][r][0] >= 0):
                        self.active[key][layer][r][0] = False
                    # Remark: Case 3 "for sure in the t range of the UNN-activation function" (given IA Bounds on Neurons): NOT POSSIBLE

                self.y.update(
                    {(i, layer, r): self.Mip.binary_var(name="y({},{})_{}".format(i, layer, r)) for r in
                     range(0, R) if self.active[key][layer][r][0]})  # binary variables for activation function
                self.eta.update(
                    {(i, layer, r): self.Mip.continuous_var(name="eta({},{})_{}".format(i, layer, r)) for r in
                     range(0, R) if self.active[key][layer][r][0]})  # slack variables
                self.mu.update(
                    {(i, layer, r): self.Mip.binary_var(name="mu({},{})_{}".format(i, layer, r)) for r in
                     range(0, R) if self.active[key][layer][r][0]})  # binary variables for activation function
                # add constraints
                for r in range(0, R):
                    if verbose is True:
                        logging.debug('Row: %s', r)
                        logging.debug('W[r,]: %s', W[r, :])
                        logging.debug('b[r]: %s', b[r])
                        logging.debug('upper (preactivated) z-bound: {}, {}, {}, {}'.format(key, layer, r,
                                                                                            self.upper_bounds_z[key][
                                                                                                layer][r][0]))
                        logging.debug('lower (preactivated) z-bound: {}, {}, {}, {}'.format(key, layer, r,
                                                                                            self.lower_bounds_z[key][
                                                                                                layer][r][0]))
                        logging.debug('L1: {}, {}, {}, {}'.format(key, layer, r, self.L1[key][layer][r]))
                        logging.debug('L2: {}, {}, {}, {}'.format(key, layer, r, self.L2[key][layer][r]))
                        logging.debug('L3: {}, {}, {}, {}'.format(key, layer, r, self.L3[key][layer][r]))
                        logging.debug('L4: {}, {}, {}, {}'.format(key, layer, r, self.L4[key][layer][r]))

                    # Case 1 "for sure in the zero range of the UNN-activation function" (given IA Bounds on Neurons)
                    if (self.upper_bounds_z[key][layer][r][0] <= 0):
                        if verbose is True:
                            logging.debug(
                                'Case 1: upper (preactivated) z-bound: {}, {}, {} is <=0 => add z==0 constraints'.format(
                                    key, layer, r))
                        self.Mip.add_constraint(ct=self.z[(i, layer, r)] == 0,
                                                ctname="Case1_CT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                        self.removed_activation_cts += 1
                    # Case 2: "for sure in the linear range of the UNN-activation function" (given IA Bounds on Neurons)
                    elif (self.upper_bounds_z[key][layer][r][0] > 0 and self.upper_bounds_z[key][layer][r][0] <= ts[
                        layer - 1] and self.lower_bounds_z[key][layer][r][0] < ts[layer - 1] and
                          self.lower_bounds_z[key][layer][r][0] >= 0):
                        if verbose is True:
                            logging.debug(
                                'Case 2: upper (preactivated) z-bound is in (0,t] AND lower (preactivated) z-bound is in [0,t): {}, {}, {} => add z==Wz_pre + b constraints'.format(
                                    key, layer, r))
                        self.Mip.add_constraint(ct=(
                                self.Mip.sum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r] ==
                                self.z[(i, layer, r)]), ctname="Case2_CT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                        self.removed_activation_cts += 1
                    # Remark: Case 3 "for sure in the t range of the UNN-activation function" (given IA Bounds on Neurons): NOT POSSIBLE

                    # Case 4: "non-decidable"
                    else:
                        # (i.)
                        self.Mip.add_constraint(
                            ct=(self.Mip.sum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r] <=
                                self.eta[(i, layer, r)]),
                            ctname="AffineCT_Bidder{}_Layer{}_Row{}_eta_i_lb".format(i, layer, r))
                        self.Mip.add_constraint(
                            ct=(self.eta[(i, layer, r)] <= self.Mip.sum(
                                W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r] + self.y[
                                    (i, layer, r)] * self.L1[key][layer][r][0]),
                            ctname="AffineCT_Bidder{}_Layer{}_Row{}_eta_i_ub".format(i, layer, r))

                        # (ii.)
                        self.Mip.add_constraint(
                            ct=(0 <= self.eta[(i, layer, r)]),
                            ctname="AffineCT_Bidder{}_Layer{}_Row{}_eta_ii_lb".format(i, layer, r))
                        self.Mip.add_constraint(
                            ct=(self.eta[(i, layer, r)] <= (1 - self.y[
                                (i, layer, r)]) * self.L2[key][layer][r][0]),
                            ctname="AffineCT_Bidder{}_Layer{}_Row{}_eta_ii_ub".format(i, layer, r))

                        # (iii.)
                        self.Mip.add_constraint(
                            ct=self.eta[(i, layer, r)] - self.mu[(i, layer, r)] * self.L3[key][layer][r][
                                0] <= self.z[(i, layer, r)],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_z_iii_lb".format(i, layer, r))
                        self.Mip.add_constraint(
                            ct=self.z[(i, layer, r)] <= self.eta[(i, layer, r)],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_z_iii_up".format(i, layer, r))

                        # (iv.)
                        self.Mip.add_constraint(
                            ct=ts[layer - 1] - (1 - self.mu[(i, layer, r)]) * self.L4[key][layer][r][0] <=
                               self.z[(i, layer, r)],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_z_iv_lb".format(i, layer, r))
                        self.Mip.add_constraint(
                            ct=self.z[(i, layer, r)] <= ts[layer - 1],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_z_iv_up".format(i, layer, r))
                layer += 1

            # Final output layer of the model
            W = self._get_model_weights(key, layer_type=['output_layer'])[0]
            R, J = W.shape
            self.z.update(
                {(i, layer, r): self.Mip.sum(
                    W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) for r in
                    range(0, R)})  # output value variables after activation

        ###################
        # 1) PLAIN NN MIP #
        ###################
        else:
            # Wb = self._clean_weights(self._get_model_weights(key))
            Wb = self._get_model_weights(key, layer_type=['layers', 'output_layer'])
            assert len(Wb) % 2 == 0, 'Currently weights and biases should always be coupled.'
            for v in range(0, len(Wb), 2):  # loop over layers
                if verbose is True:
                    logging.debug('\nLayer: %s', layer)
                W = Wb[v]
                if verbose is True:
                    logging.debug('W: %s', W.shape)
                b = Wb[v + 1]
                # Negate the bias like in the layer
                if verbose is True:
                    logging.debug('b: %s', b.shape)
                R, J = W.shape
                # decision variables
                if v == 0:
                    self.z.update(
                        {(i, 0, j): self.Mip.binary_var(name="x({})_{}".format(i, j)) for j in
                         range(0, J)})  # binary variables for allocation
                self.z.update(
                    {(i, layer, r): self.Mip.continuous_var(lb=0, name="z({},{})_{}".format(i, layer, r)) for r in
                     range(0, R)})  # output value variables after activation
                self.s.update(
                    {(i, layer, r): self.Mip.continuous_var(lb=0, name="s({},{})_{}".format(i, layer, r)) for r in
                     range(0, R) if
                     (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][
                         0] != 0)})  # slack variables
                self.y.update(
                    {(i, layer, r): self.Mip.binary_var(name="y({},{})_{}".format(i, layer, r)) for r in range(0, R) if
                     (self.upper_bounds_z[key][layer][r][0] != 0 and self.upper_bounds_s[key][layer][r][
                         0] != 0)})  # binary variables for activation function

                # Slack Formulation: add constraints
                for r in range(0, R):

                    if verbose is True:
                        logging.debug('Row: %s', r)
                        logging.debug('W[r,]: %s', W[r, :])
                        logging.debug('b[r]: %s', b[r])
                        logging.debug(
                            '(preactivated) upper z-bound: {}, {}, {}, {}'.format(key, layer, r,
                                                                                  self.upper_bounds_z[key][layer][r]))
                        logging.debug(
                            '(preactivated) upper s-bound: {}, {}, {}, {}'.format(key, layer, r,
                                                                                  self.upper_bounds_s[key][layer][r]))

                    # Case 1: "for sure in the zero range of the relu" (given IA Bounds on Neurons)
                    if self.upper_bounds_z[key][layer][r][0] == 0:
                        if verbose is True:
                            logging.debug(
                                'Case 1: upper z-bound: {}, {}, {} is equal to zero => add z==0 constraints'.format(key,
                                                                                                                    layer,
                                                                                                                    r))
                        self.Mip.add_constraint(ct=self.z[(i, layer, r)] == 0,
                                                ctname="Case1_CT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                        self.removed_activation_cts += 1


                    # Case 2: "for sure in the linear range of the relu" (given IA Bounds on Neurons)
                    elif self.upper_bounds_s[key][layer][r][0] == 0:
                        if verbose is True:
                            logging.debug(
                                'Case 2: upper s-bound: {}, {}, {} is equal to zero => add z==Wz_pre + b constraints'.format(
                                    key, layer, r))
                        self.Mip.add_constraint(ct=(
                                self.Mip.sum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r] ==
                                self.z[(i, layer, r)]), ctname="Case2_CT_Bidder{}_Layer{}_Row{}".format(i, layer, r))
                        self.removed_activation_cts += 1

                    # Case 3: "not decidable for sure" (given IA Bounds on Neurons) => need to encode full relu with slack
                    else:
                        # (i) affine constraint
                        self.Mip.add_constraint(ct=(
                                self.Mip.sum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r] ==
                                self.z[(i, layer, r)] - self.s[(i, layer, r)]),
                            ctname="AffineCT_Bidder{}_Layer{}_Row{}".format(i, layer, r))

                        # (ii) big-M constraint
                        self.Mip.add_constraint(
                            ct=self.z[(i, layer, r)] <= self.y[(i, layer, r)] * self.upper_bounds_z[key][layer][r][0],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_z".format(i, layer, r))

                        # (iii) big-M constraint
                        self.Mip.add_constraint(
                            ct=self.s[(i, layer, r)] <= (1 - self.y[(i, layer, r)]) *
                               self.upper_bounds_s[key][layer][r][0],
                            ctname="BinaryCT_Bidder{}_Layer{}_Row{}_s".format(i, layer, r))

                    if verbose is True:
                        for m in range(0, self.Mip.number_of_constraints):
                            if self.Mip.get_constraint_by_index(m) is not None:
                                logging.debug(self.Mip.get_constraint_by_index(m))

                layer = layer + 1

    def initialize_mip(self,
                       verbose=False,
                       bidder_specific_constraints=None,
                       GSVM_specific_constraints=False,
                       national_circle_complement=None):
        # pay attention here order is important, thus first sort the keys of bidders!
        logging.info('')
        logging.info('Initialize MIP')
        logging.info('-----------------------------------------------')
        logging.debug('Sorted active bidders in MIP: %s', self.sorted_bidders)
        # linear matrix constraints: Wz^(i-1)+b = z^(i)-s^(i)
        for i in range(0, self.N):
            self._add_matrix_constraints(i, verbose=verbose)
        # allocation constraints for x^i's
        for j in range(0, self.M):
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, 0, j)] for i in range(0, self.N)) <= 1),
                                    ctname="FeasabilityCT_x_{}".format(j))
        # add bidder specific constraints
        if bidder_specific_constraints is not None:
            self._add_bidder_specific_constraints(bidder_specific_constraints)

        #  GSVM specific allocation constraints for regional and local bidder
        if GSVM_specific_constraints and national_circle_complement is not None:
            for i in range(0, self.N):
                # regional bidder
                if self.sorted_bidders[i] in ['Bidder_0', 'Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']:
                    logging.debug('Adding GSVM specific constraints for regional {}.'.format(self.sorted_bidders[i]))
                    self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, 0, j)] for j in range(0, self.M)) <= 4),
                                            ctname="GSVM_CT_RegionalBidder{}".format(i))
                # national bidder
                elif self.sorted_bidders[i] in ['Bidder_6']:
                    logging.debug(
                        'Adding GSVM specific constraints for national {} with national circle complement {}.'.format(
                            self.sorted_bidders[i], national_circle_complement))
                    self.Mip.add_constraint(
                        ct=(self.Mip.sum(self.z[(i, 0, j)] for j in national_circle_complement) == 0),
                        ctname="GSVM_CT_NationalBidder{}".format(i))
                else:
                    raise NotImplementedError(
                        'GSVM only implmented in default version for Regional Bidders:[Bidder_0,..,Bidder_5] and National Bidder: [Bidder_6]. You entered {}'.format(
                            self.sorted_bidders[i]))

        # add objective: sum of 1dim outputs of neural network per bidder z[(i,K_i,0)]
        objective = self.Mip.sum(
            self.Models[self.sorted_bidders[i]]._target_max * self.z[
                (i, self.Models[self.sorted_bidders[i]]._num_hidden_layers + 1, 0)] for i in range(0, self.N))
        self.Mip.maximize(objective)
        logging.info('MIP initialized')

    def _add_bidder_specific_constraints(self,
                                         bidder_specific_constraints):
        for bidder_key, bundles in bidder_specific_constraints.items():
            bidder_id = np.where([x == bidder_key for x in self.sorted_bidders])[0][0]
            logging.debug('Adding bidder specific constraints')
            for idx, bundle in enumerate(bundles):
                # logging.debug(bundle)
                self.Mip.add_constraint(
                    ct=(self.Mip.sum((self.z[(bidder_id, 0, j)] == bundle[j]) for j in range(0, self.M)) <= self.M - 1),
                    ctname="BidderSpecificCT_Bidder{}_No{}".format(bidder_id, idx))

    def get_bidder_key_position(self,
                                bidder_key):
        return np.where([x == bidder_key for x in self.sorted_bidders])[0][0]

    def reset_mip(self):
        self.Mip = cpx.Model(name="MIP")

    def tighten_bounds_LP(self,
                          upper_bound_input,
                          lower_bound_input,
                          verbose=False):
        raise NotImplementedError('LP-tightening not implemented')

    def tighten_bounds_IA(self,
                          upper_bound_input,
                          lower_bound_input,
                          verbose=False):
        logging.info('')
        logging.info('Tighten Bounds with IA')
        logging.info('-----------------------------------------------')

        ##############
        # 1) UNN MIP #
        ##############
        if self.NN_type == 'UNN':
            phi = lambda x, t: np.minimum(t, np.maximum(0, x))

            for bidder in self.sorted_bidders:
                logging.debug('Tighten bounds with IA for %s', bidder)
                Wb = self._get_model_weights(bidder)
                ts = self.Models[bidder].ts
                assert len(Wb) % 2 == 0, 'Currently weights and biases should always be coupled.'

                #######
                # Initializing the absolute upper and lower bound on the input
                self.upper_bounds_z[bidder][0] = np.array(upper_bound_input).reshape(-1, 1)
                self.lower_bounds_z[bidder][0] = np.array(lower_bound_input).reshape(-1, 1)
                #######

                layer = 1
                for v in range(0, len(Wb), 2):  # loop over layers
                    W = Wb[v]
                    b = Wb[v + 1]
                    R, J = W.shape

                    ######
                    # (preactivated) upper/lower bounds
                    self.upper_bounds_z[bidder][layer] = W @ phi(self.upper_bounds_z[bidder][layer - 1],
                                                                 ts[layer - 1]) + b.reshape(-1, 1)  # upper bound for z
                    self.lower_bounds_z[bidder][layer] = W @ phi(self.lower_bounds_z[bidder][layer - 1],
                                                                 ts[layer - 1]) + b.reshape(-1, 1)  # lower bound for z
                    #####

                    ######
                    # -> Set the from the upper/lower bounds implied big-M variables
                    for r in range(0, R):
                        # if the (preactivated) upper/lower bound is outside [0,t], then project it back for the calculation of L1,L2,L3,L4
                        self.L1[bidder][layer][r][0] = max(0, - b[r])
                        self.L2[bidder][layer][r][0] = max(0, sum(
                            W[r, j] * phi(self.upper_bounds_z[bidder][layer - 1][j], ts[layer - 1]) for j in
                            range(0, J)) + b[r])
                        self.L3[bidder][layer][r][0] = max(0, self.L2[bidder][layer][r][0] - ts[layer - 1])
                        self.L4[bidder][layer][r][0] = ts[layer - 1]
                    ######

                    layer += 1

            if verbose is True:
                logging.debug('(preactivated) Upper Bounds z:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')
                logging.debug('(preactivated) Lower Bounds z:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.lower_bounds_z.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')
                logging.debug('L1:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.L1.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')
                logging.debug('L2:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.L2.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')
                logging.debug('L3:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.L3.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')
                logging.debug('L4:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.L4.items()}).fillna('-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('')

        ###################
        # 2) PLAIN NN MIP #
        ###################
        else:
            for bidder in self.sorted_bidders:
                logging.debug('Tighten bounds with IA for PlainNN %s', bidder)
                Wb = self._get_model_weights(bidder, layer_type=['layers', 'output_layer'])

                #######
                # Initializing the absolute upper and lower bound on the input
                self.upper_bounds_z[bidder][0] = np.array(upper_bound_input).reshape(-1, 1)
                self.upper_bounds_s[bidder][0] = np.array(upper_bound_input).reshape(-1, 1)
                #######

                layer = 1
                for v in range(0, len(Wb), 2):  # loop over layers
                    W = Wb[v]
                    b = Wb[v + 1]
                    W_plus = np.maximum(W, 0)
                    W_minus = np.minimum(W, 0)

                    ######
                    # (preactivated) upper/lower bounds
                    self.upper_bounds_z[bidder][layer] = np.ceil(
                        np.maximum(W_plus @ self.upper_bounds_z[bidder][layer - 1] + b.reshape(-1, 1),
                                   0)).astype(int)  # upper bound for z
                    self.upper_bounds_s[bidder][layer] = np.ceil(
                        np.maximum(-(W_minus @ self.upper_bounds_z[bidder][layer - 1] + b.reshape(-1, 1)),
                                   0)).astype(int)  # upper bound  for s
                    ######

                    layer += 1

            if verbose is True:
                logging.debug('(preactivated) Upper Bounds z:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna(
                        '-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())
                logging.debug('(preactivated) Upper Bounds s:')
                for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna(
                        '-').items():
                    logging.debug(k)
                    logging.debug(v.to_string())

    def print_bounds(self):
        print('\n################################ BOUNDS ################################')
        ##############
        # 1) UNN MIP #
        ##############
        if self.NN_type == 'UNN':
            print('(preactivated) Upper Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                print(k)
                print(v.to_string())
            print('')
            print('(preactivated) Lower Bounds z:')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.lower_bounds_z.items()}).fillna('-').items():
                print(k)
                print(v.to_string())
            print(f'Number of UNN-activation-function-constraints removed: {self.removed_activation_cts}')

        ###################
        # 2) PLAIN NN MIP #
        ###################
        else:
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_z.items()}).fillna('-').items():
                print('(preactivated) Upper Bounds z:')
                print(k)
                print(v.to_string())
            print('')
            for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_bounds_s.items()}).fillna('-').items():
                print('(preactivated) Upper Bounds s:')
                print(k)
                print(v.to_string())
            print(f'Number of ReLu-constraints removed: {self.removed_activation_cts}')
        print('#########################################################################')
        print('\n')


# %%
print('Class NN_MIP_TORCH imported')
