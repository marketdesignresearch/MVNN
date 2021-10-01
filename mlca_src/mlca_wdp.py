import logging

# CPLEX: Here, DOcplex is used for solving the deep neural network-based Winner Determination Problem.
import docplex.mp.model as cpx
# Libs
import numpy as np
import pandas as pd
# %%


class MLCA_WDP:

    def __init__(self, bids):

        self.bids = bids  # list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
        self.N = len(bids)  # number of bidders
        self.M = bids[0].shape[1] - 1  # number of items
        self.Mip = cpx.Model(name="WDP")  # cplex model
        self.K = [x.shape[0] for x in bids]  # number of elicited bids per bidder
        self.z = {}  # decision variables. z(i,k) = 1 <=> bidder i gets the kth bundle out of 1,...,K[i] from his set of bundle-value pairs
        self.x_star = np.zeros((self.N, self.M), dtype=int)  # optimal allocation of the winner determination problem

    def initialize_mip(self, verbose=0):

        for i in range(0, self.N):  # over bidders i \in N
            # add decision variables
            self.z.update({(i, k): self.Mip.binary_var(name="z({},{})".format(i, k)) for k in
                           range(0, self.K[i])})  # z(i,k) = 1 <=> bidder i gets bundle k \in Ki
            # add allocation constraints for z(i,k)
            self.Mip.add_constraint(ct=(self.Mip.sum(self.z[(i, k)] for k in range(self.K[i])) <= 1),
                                    ctname="CT Allocation Bidder {}".format(i))

        # add intersection constraints of buzndles for z(i,k)
        for m in range(0, self.M):  # over items m \in M
            self.Mip.add_constraint(ct=(self.Mip.sum(
                self.z[(i, k)] * self.bids[i][k, m] for i in range(0, self.N) for k in range(0, self.K[i])) <= 1),
                                    ctname="CT Intersection Item {}".format(m))

        # add objective
        objective = self.Mip.sum(
            self.z[(i, k)] * self.bids[i][k, self.M] for i in range(0, self.N) for k in range(0, self.K[i]))
        self.Mip.maximize(objective)

        if verbose == 1:
            for m in range(0, self.Mip.number_of_constraints):
                logging.debug('({}) %s'.format(m), self.Mip.get_constraint_by_index(m))
            logging.debug('\nMip initialized')

    def solve_mip(self, verbose=0):
        self.Mip.solve()
        if verbose == 1:
            self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for k in range(0, self.K[i]):
                if self.z[(i, k)].solution_value != 0:
                    self.x_star[i, :] = self.z[(i, k)].solution_value * self.bids[i][k, :-1]

    def log_solve_details(self, solved_mip):
        details = solved_mip.get_solve_details()
        logging.debug('Status  : %s', details.status)
        logging.debug('Time    : %s sec', round(details.time))
        logging.debug('Problem : %s', details.problem_type)
        logging.debug('Rel. Gap: {} %'.format(details.mip_relative_gap))
        logging.debug('N. Iter : %s', details.nb_iterations)
        logging.debug('Hit Lim.: %s', details.has_hit_limit())
        logging.debug('Objective Value: %s', solved_mip.objective_value)

    def summary(self):
        print('################################ OBJECTIVE ################################')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics(), '\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('########################### ALLOCATED BIDDERs ############################')
        try:
            for i in range(0, self.N):
                for k in range(0, self.K[i]):
                    if self.z[(i, k)].solution_value != 0:
                        print('z({},{})='.format(i, k), int(self.z[(i, k)].solution_value))
        except Exception:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ###############################')
        self.print_optimal_allocation()
        return (' ')

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M + 1)]
        print(D)
        print('\nItems allocated:')
        print(D.sum(axis=0))


print('Class MLCA_WDP imported')
