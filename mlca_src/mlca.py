import json
import logging
from datetime import datetime

import numpy as np
from numpyencoder import NumpyEncoder

from mlca_src.mlca_economies import MLCA_Economies
from mlca_src.mlca_util import key_to_int, timediff_d_h_m_s
from sats.pysats import PySats

# %% MLCA MECHANISM
def mlca_mechanism(SATS_auction_instance_seed=None, SATS_domain_name=None, Qinit=None, Qmax=None, Qround=None,
                   local_scaling_factor=None, NN_parameters=None, MIP_parameters=None, scaler=None,
                   init_bids_and_fitted_scaler=[None, None], return_allocation=False, return_payments=False,
                   calc_efficiency_per_iteration=False, isLegacy=False, configdict=None, res_path=None):
    start = datetime.now()
    # can also specify input as a dict of an configuration
    if configdict is not None:
        SATS_domain_name = configdict['SATS_domain_name']
        SATS_auction_instance_seed = configdict['SATS_auction_instance_seed']
        Qinit = configdict['Qinit']
        Qmax = configdict['Qmax']
        Qround = configdict['Qround']
        NN_parameters = configdict['NN_parameters']
        MIP_parameters = configdict['MIP_parameters']
        scaler = configdict['scaler']
        init_bids_and_fitted_scaler = configdict['init_bids_and_fitted_scaler']
        return_allocation = configdict['return_allocation']
        return_payments = configdict['return_payments']
        calc_efficiency_per_iteration = configdict['calc_efficiency_per_iteration']
        isLegacy = configdict['isLegacy']

    logging.warning('START MLCA:')
    logging.warning('-----------------------------------------------')
    OUTPUT = {}
    logging.warning('Model: %s', SATS_domain_name)
    logging.warning('Seed SATS Instance: %s', SATS_auction_instance_seed)
    logging.warning('Qinit: %s', Qinit) if (init_bids_and_fitted_scaler[0] is None) else \
        logging.warning('Qinit: %s', [v[0].shape[0] for k, v in init_bids_and_fitted_scaler[0].items()])
    logging.warning('Qmax: %s', Qmax)
    logging.warning('Qround: %s\n', Qround)

    # Instantiate Economies
    logging.warning('Instantiate SATS Instance')
    if SATS_domain_name == 'LSVM':
        SATS_auction_instance = PySats.getInstance().create_lsvm(seed=SATS_auction_instance_seed,
                                                                 isLegacyLSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyLSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
    if SATS_domain_name == 'GSVM':
        SATS_auction_instance = PySats.getInstance().create_gsvm(seed=SATS_auction_instance_seed,
                                                                 isLegacyGSVM=isLegacy)  # create SATS auction instance
        logging.warning('####### ATTENTION #######')
        logging.warning('isLegacyGSVM: %s', SATS_auction_instance.isLegacy)
        logging.warning('#########################\n')
    if SATS_domain_name == 'MRVM':
        SATS_auction_instance = PySats.getInstance().create_mrvm(
            seed=SATS_auction_instance_seed)  # create SATS auction instance
    if SATS_domain_name == 'SRVM':
        SATS_auction_instance = PySats.getInstance().create_srvm(
            seed=SATS_auction_instance_seed)  # create SATS auction instance
    E = MLCA_Economies(SATS_auction_instance=SATS_auction_instance,
                       SATS_auction_instance_seed=SATS_auction_instance_seed,
                       Qinit=Qinit, Qmax=Qmax, Qround=Qround, scaler=scaler,
                       local_scaling_factor=local_scaling_factor)  # create economy instance

    E.set_NN_parameters(parameters=NN_parameters)  # set NN parameters
    E.set_MIP_parameters(parameters=MIP_parameters)  # set MIP parameters

    # Set initial bids | Line 1-3
    init_bids, init_fitted_scaler = init_bids_and_fitted_scaler
    if init_bids is not None:
        E.set_initial_bids(initial_bids=init_bids,
                           fitted_scaler=init_fitted_scaler)  # (*) use self defined inital bids | Line 1
    else:
        E.set_initial_bids(
            seed=SATS_auction_instance_seed)  # create inital bids B^0, uniformly sampling at random from bundle space | Line 2 (now depending on isLegacy with correct sampling)

    # Calculate efficient allocation given current elicited bids
    if calc_efficiency_per_iteration:
        E.calculate_efficiency_per_iteration()

    # Global while loop: check if for all bidders one addtitional auction round is feasible | Line 4
    Rmax = max(E.get_number_of_elicited_bids().values())
    CHECK = Rmax <= (E.Qmax - E.Qround)
    while CHECK:

        # Increment iteration
        E.mlca_iteration += 1
        # log info
        E.get_info()

        # Reset Attributes | Line 18
        logging.info('RESET: Auction Round Query Profile S=(S_1,...,S_n)')
        E.reset_current_query_profile()
        logging.info('RESET: Status of Economies')
        E.reset_economy_status()
        logging.info('RESET: NN Models')
        E.reset_NN_models()
        logging.info('RESET: Argmax Allocation\n')
        E.reset_argmax_allocations()

        # Check Current Query Profile: | Line 5
        logging.debug('Current query profile S=(S_1,...,S_n):')
        for k, v in E.current_query_profile.items():
            logging.debug(k + ':  %s', v)

        # TODO: fit NN only once in each iteration as in mlca_mechanism_ft?

        # Marginal Economies: | Line 6-12
        logging.info('MARGINAL ECONOMIES FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')
        for bidder in E.bidder_names:
            logging.info(bidder)
            logging.info('-----------------------------------------------')
            logging.debug('Sampling marginals for %s', bidder)
            sampled_marginal_economies = E.sample_marginal_economies(active_bidder=key_to_int(bidder),
                                                                     number_of_marginals=E.Qround - 1)
            sampled_marginal_economies.sort()
            logging.debug('Calculate next queries for the following sampled marginals:')
            for marginal_economy in sampled_marginal_economies:
                logging.debug(marginal_economy)

            for marginal_economy in sampled_marginal_economies:
                logging.debug('')
                logging.info(bidder + ' | ' + marginal_economy)
                logging.info('-----------------------------------------------')
                logging.info('Status of Economy: %s\n', E.economy_status[marginal_economy])
                q_i = E.next_queries(economy_key=marginal_economy, active_bidder=bidder)
                E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
                logging.debug('')
                logging.debug('Current query profile for %s:', bidder)
                for k in range(E.current_query_profile[bidder].shape[0]):
                    logging.debug(E.current_query_profile[bidder][k, :])
                logging.debug('')

        # For increasing speed clear tf models of marginal economies
        logging.info('RESET: NN Models of marginal Economies')
        E.reset_NN_models()

        # Main Economy: | Line 13-14
        logging.info('MAIN ECONOMY FOR ALL BIDDERS')
        logging.info('-----------------------------------------------\n')
        for bidder in E.bidder_names:
            logging.info(bidder)
            logging.info('-----------------------------------------------')
            economy_key = 'Main Economy'
            logging.debug(economy_key)
            logging.debug('-----------------------------------------------')
            logging.debug('Status of Economy: %s', E.economy_status[economy_key])
            q_i = E.next_queries(economy_key=economy_key, active_bidder=bidder)
            E.update_current_query_profile(bidder=bidder, bundle_to_add=q_i)
            logging.debug('')
            logging.debug('Current query profile for %s:', bidder)
            for k in range(E.current_query_profile[bidder].shape[0]):
                logging.debug(E.current_query_profile[bidder][k, :])
            logging.debug('')

        # Update Elicited Bids With Current Query Profile and check uniqueness | Line 15-16
        if not E.update_elicited_bids():
            logging.warning('UNIQUENESS CHECK FAILED, STOP MLCA!')
            return

        # Calculate efficient allocation given current elicited bids
        if calc_efficiency_per_iteration:
            efficiency = E.calculate_efficiency_per_iteration()
            if np.isclose(efficiency, 1.0, rtol=1e-4):
                logging.info('EARLY STOPPING - 100% efficiency reached.')
                break

        # TODO:Eff check/break condition as in mlca_mechanism_ft file
        if res_path is not None:
            json.dump(
                {'Statistics': {'Elapsed Times of MIPs': E.elapsed_time_mip,
                                'NN Losses': E.losses, 'Efficiency per Iteration': E.efficiency_per_iteration,
                                'Efficient Allocation per Iteration': E.efficient_allocation_per_iteration,
                                'Relative Revenue': E.relative_revenue, 'mip_statistics': E.mip_log,
                                'nn_train_logs': E.train_logs},
                 'MLCA Payments': E.mlca_payments,
                 'MLCA Allocation': E.mlca_allocation,
                 'Elicited Bids': E.elicited_bids},
                open(res_path, 'w'),
                indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)

        # Update while condition
        Rmax = max(E.get_number_of_elicited_bids().values())
        CHECK = Rmax <= (E.Qmax - E.Qround)

    # allocation & payments # | Line 20
    logging.info('ALLOCATION')
    logging.info('---------------------------------------------')
    E.calculate_mlca_allocation()
    E.mlca_allocation_efficiency = E.calculate_efficiency_of_allocation(E.mlca_allocation, E.mlca_scw, verbose=1)
    if return_payments:  # | Line 21
        logging.info('')
        logging.info('PAYMENTS')
        logging.info('---------------------------------------------')
        E.calculate_vcg_payments()

    end = datetime.now()
    total_time_elapsed = timediff_d_h_m_s(end - start)
    E.get_info(final_summary=True)
    logging.warning('NORMAL OPTIMIZATIONS: {}/ BIDDER SPECIFIC: {}'.format(
        E.number_of_optimization['normal'], E.number_of_optimization['bidder_specific']))
    logging.warning('EFFICIENCY: {} %'.format(round(E.mlca_allocation_efficiency, 4) * 100))
    logging.warning('TOTAL TIME ELAPSED: {}'.format('{}d {}h:{}m:{}s'.format(*total_time_elapsed)))
    logging.warning('MLCA FINISHED')

    # Set OUTPUT
    OUTPUT['MLCA Efficiency'] = E.mlca_allocation_efficiency
    if return_allocation: OUTPUT['MLCA Allocation'] = E.mlca_allocation
    OUTPUT['MLCA Payments'] = E.mlca_payments
    OUTPUT['Statistics'] = {'Total Time Elapsed': total_time_elapsed, 'Elapsed Times of MIPs': E.elapsed_time_mip,
                            'NN Losses': E.losses, 'Efficiency per Iteration': E.efficiency_per_iteration,
                            'Efficient Allocation per Iteration': E.efficient_allocation_per_iteration,
                            'Relative Revenue': E.relative_revenue, 'mip_statistics': E.mip_log,
                            'nn_train_logs': E.train_logs}
    OUTPUT['Elicited Bids'] = E.elicited_bids
    return (SATS_auction_instance_seed, OUTPUT)


# %%
print('FUNCTION MLCA imported')
