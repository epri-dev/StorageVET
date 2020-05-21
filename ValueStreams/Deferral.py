"""
Deferral.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Halley Nathwani, Miles Evans, Kunle Awojinrin and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Thien Nguyen', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import numpy as np
import cvxpy as cvx
from .ValueStream import ValueStream
import pandas as pd
import Library as Lib
import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class Deferral(ValueStream):
    """ Investment deferral. Each service will be daughters of the PreDispService class.

    """

    def __init__(self, params, techs, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            dt (float): optimization timestep (hours)
        """

        # generate the generic service object
        ValueStream.__init__(self, techs['Storage'], 'Deferral', dt)

        # add Deferral specific attributes
        self.max_import = params['planned_load_limit']  # positive
        self.max_export = params['reverse_power_flow_limit']  # negative
        self.last_year = params['last_year'].year
        self.year_failed = params['last_year'].year + 1
        self.min_power_requirement = 0
        self.min_energy_requirement = 0
        self.load = params['load']  # deferral load
        self.growth = params['growth']  # Growth Rate of deferral load (%/yr)
        self.price = params['price']  # $/yr

        self.verbose = False  # set to true if you want to debug this value stream  --HN
        self.deferral_df = None
        # self.included_load = params['included_load']  # deferral load has already been add to the aggregation of loads

    def check_for_deferral_failure(self, end_year, technologies, frequency, opt_years):
        """This functions checks the constraints of the storage system against any predispatch or user inputted constraints
        for any infeasible constraints on the system.

        The goal of this function is to predict the year that storage will fail to deferral a T&D asset upgrade.

        Only runs if Deferral is active.

        Args:
            end_year:
            technologies:
            frequency:
            opt_years:

        Returns: new list of optimziation years

        """
        u_logger.info('Finding first year of deferral failure...')
        current_year = self.load.index.year[-1]

        additional_years = [current_year]
        already_failed = False

        tech = technologies['Storage']
        rte = tech.rte
        max_ch = tech.ch_max_rated
        max_dis = tech.dis_max_rated
        max_ene = tech.ene_max_rated * tech.ulsoc
        years_deferral_column = []
        min_power_deferral_column = []
        min_energy_deferral_column = []

        while current_year <= end_year.year:
            size = len(self.load)
            print('current year: ' + str(current_year)) if self.verbose else None
            years_deferral_column.append(current_year)

            # TODO can we check the max year? or can we take the previous answer to calculate the next energy requirements?
            generation = np.zeros(size)
            if 'ICE' in technologies.keys():
                # diesel generation is constant over the year, so add it out side the loop
                diesel = technologies['ICE']
                generation += np.repeat(diesel.rated_power*diesel.n, size)
            if 'PV' in technologies.keys():
                generation += technologies['PV'].generation.values
            load = np.zeros(size)
            if 'Load' in technologies.keys():
                load += technologies['Load'].site_load.values
            min_power, min_energy = self.precheck_failure(self.dt, rte, load, generation)
            print(f'In {current_year} -- min power: {min_power}  min energy: {min_energy }') if self.verbose else None
            min_power_deferral_column.append(min_power)
            min_energy_deferral_column.append(min_energy)

            if not already_failed and (min_power > max_ch or min_power > max_dis or min_energy > max_ene):
                # then we predict that deferral will fail
                last_deferral_yr = current_year - 1
                self.set_last_deferral_year(last_deferral_yr, current_year)

                opt_years = list(set(opt_years + additional_years))
                already_failed = True
                u_logger.info(f'Running analysis on years: {opt_years}')

            # the current year we have could be the last year the deferral is possible, so we want
            # to keep it in self.opt_results until we know the next is can be deferred as well
            additional_years = [current_year, current_year + 1]
            next_opt_years = list(set(opt_years + additional_years))

            # add additional year of data to loads profiles
            if 'Load' in technologies.keys():
                technologies['Load'].estimate_year_data(next_opt_years, frequency)

            # add additional year of PV generation forecast
            if 'PV' in technologies.keys():
                technologies['PV'].estimate_year_data(next_opt_years, frequency)

            # add additional year of data to deferred load
            self.estimate_year_data(next_opt_years, frequency)

            # index the current year by one
            current_year += 1

        self.deferral_df = pd.DataFrame({'Year': years_deferral_column,
                                         'Power Capacity Requirement (kW)': min_power_deferral_column,
                                         'Energy Capacity Requirement (kWh)': min_energy_deferral_column})
        self.deferral_df.set_index('Year', inplace=True)
        return opt_years

    def precheck_failure(self, tstep, rte, sys_load, generation):
        """
        This function takes in a vector of storage power requirements (negative=charging and positive=discharging) [=] kW
        that are required to perform the deferral as well as a time step (tstep) [=] hrs

        Args:
            tstep (float): timestep of the data in hours
            rte (float): round trip efficiency of storage TODO -- what should this value be with multiple storage
            sys_load (Series): Series or DataFrame of total load
            generation (list, ndarray): total generation

        Returns:
            A list of minimum storage power (kw_rated) and energy (kwh_min) capacity required to meet deferral

        Notes:
            This algorithm can reliably find the last year deferral is possible, however the problem might still
            be found INFEASIBLE if the ESS cannot use it's full range of SOC (ie. if LLSOC is too high or ULSOC is too low)
        """
        load = self.load.values + sys_load
        net_feeder_load = load - generation

        # Determine power requirement of the storage:
        # (1) anytime the net_feeder_load goes above deferral_max_import
        positive_feeder_load = net_feeder_load.clip(min=0)
        positive_load_power_req = positive_feeder_load - self.max_import
        positive_power_req = positive_load_power_req.clip(min=0)
        # (2) anytime the net_feeder_load goes below deferral_max_exports (assumes deferral_max_export < 0)
        negative_feeder_load = net_feeder_load.clip(max=0)
        negative_load_power_req = negative_feeder_load - self.max_export
        negative_power_req = negative_load_power_req.clip(max=0)
        # The sum of (1) and (2)
        sto_p_req = positive_power_req + negative_power_req

        # Loop through time steps. If the storage is forced to dispatch from the constraint,
        # return to nominal SOC as soon as possible after.
        kw_rated = max(abs(sto_p_req))
        sto_dispatch = np.zeros(sto_p_req.shape)
        e_walk = np.zeros(sto_p_req.shape)  # how much the energy in the ESS needs to wander #Definitely not a star wars pun
        for step in range(len(sto_p_req)):
            if step == 0:
                e_walk[step] = -tstep * sto_p_req[0]  # initialize at nominal SOC
                sto_dispatch[step] = sto_p_req[0]  # ignore constaints imposed by the first timestep of the year
            elif sto_p_req[step] > 0:  # if it is required to dispatch, do it
                sto_dispatch[step] = sto_p_req[step]
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep  # kWh
            elif sto_p_req[step] < 0:
                sto_dispatch[step] = sto_p_req[step]
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep * rte
            elif e_walk[step - 1] < 0:  # Otherwise contribute its full power to returning energy to nominal
                sto_dispatch[step] = -min(abs(kw_rated), abs(e_walk[step - 1] / tstep))
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep * rte  # kWh
            elif e_walk[step - 1] > 0:
                sto_dispatch[step] = min(abs(kw_rated), abs(e_walk[step - 1] / tstep))
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep  # kWh
            else:
                sto_dispatch[step] = 0
                e_walk[step] = e_walk[step - 1]
        kwh_min = max(e_walk) - min(e_walk)
        self.min_power_requirement = float(kw_rated)
        self.min_energy_requirement = float(kwh_min)
        return [self.min_power_requirement, self.min_energy_requirement]

    def set_last_deferral_year(self, last_year, failed_year):
        """Sets last year that deferral is possible

        Args:
            last_year (int): The last year storage can defer an T&D equipment upgrade
            failed_year (int): the year that deferring an upgrade will fail
        """
        self.last_year = last_year
        self.year_failed = failed_year
        print('year failed set to: ' + str(self.year_failed)) if self.verbose else None

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.load.index.year.unique()

        # which years is data given for that is not needed
        dont_need_year = {pd.Period(year) for year in data_year} - {pd.Period(year) for year in years}
        if len(dont_need_year) > 0:
            for yr in dont_need_year:
                sub = self.load[self.load.index.year != yr.year]  # choose all data that is not in the unneeded year
                self.load = sub

        data_year = self.load.index.year.unique()
        # which years do we not have data for
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}
        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                source_data = self.load[self.load.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.load = pd.concat([self.load, new_data], sort=True)  # add to existing

    def objective_constraints(self, variables, mask, load, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
                reservations (Dict): power reservations from dispatch services

        Returns:
            An list of constraints to be included for deferral

        # TODO: consider verifying timesteps are in sequential order before just taking the values of the Series --HN
        """

        # adding constraints to ensure power dispatch does not violate thermal limits of transformer deferred
        # only include them if deferral is not going to fail
        constraints = []
        year_of_optimization = mask.loc[mask].index.year[-1]
        print(str(year_of_optimization)) if self.verbose else None
        if year_of_optimization < self.year_failed:
            print('adding constraint') if self.verbose else None
            # optimization variables
            dis = variables['dis']
            ch = variables['ch']

            net_battery = dis - ch

            tot_load_deferred = load + self.load.loc[mask].values

            # -(max export) >= dis - ch + generation - deferral load
            constraints += [cvx.NonPos(self.max_export + generation - tot_load_deferred + net_battery)]

            # max import >= loads - (dis - ch) - generation
            constraints += [cvx.NonPos(-generation + tot_load_deferred - net_battery - self.max_import)]

            # # make sure power does doesn't go above the inverter constraints during dispatch service activity
            # constraints += [cvx.NonPos(self.max_export - load_tot + net_battery + generation + reservations['D_max'] + reservations['C_min'])]
            # constraints += [cvx.NonPos(load_tot - net_battery - generation - reservations['C_max'] - reservations['D_min'] - self.max_import)]

        return constraints

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.load.index)
        report.loc[:, 'Deferral Load (kW)'] = self.load
        return report

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream


        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided), a list (of columns that remain zero), and a list (of columns that
        retain a constant value over the entire project horizon).

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        years = results.index.year.unique()
        start_year = min(years)
        end_year = max(years)

        yr_index = pd.period_range(start=start_year, end=end_year, freq='y')

        proforma = pd.DataFrame(data={self.name + ' Value': np.zeros(len(yr_index))}, index=yr_index)

        for year in years:

            if year >= self.year_failed:
                proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Value'] = 0
            else:
                proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Value'] = self.price

        return proforma, [self.name + ' Value'], None

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:   just
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value

    def check_if_deferral_only(self, all_components_list):
        """
        This method takes in a List of components and their activation status, and checks to see if deferral is the
        only service (pre-dispatch or regular service) that is activated

        Args:
            all_components_list (List): list of Technology Input map, Pre-dispatch Service Input map, and Service Input
            map
        Returns:
            deferral_only (bool): True if deferral/battery is the only pre-dispatch service/technology pair activated,
            False otherwise
        """
        # TODO: possible error regarding size of an empty/null dict? Assuming that these maps are empty if not active

        deferral_only = False
        predispatch_services_map = all_components_list[1]
        services_map = all_components_list[2]

        valid_predispatch_services_map = {}  # predispatch_services_map stripped of all keys w/ None values
        for key, value in predispatch_services_map.items():
            if value is not None:
                valid_predispatch_services_map[key] = value

        valid_services_map = {}  # services_map stripped of all keys w/ None values
        for key, value in services_map.items():
            if value is not None:
                valid_services_map[key] = value

        # TODO: delete after pytesting
        #print("PREDISPATCH SERVICES ----->", valid_predispatch_services_map)
        #print("PREDISPATCH SERVICES LENGTH ---->", len(valid_predispatch_services_map))
        #print("SERVICES ----->", valid_services_map)
        #print("SERVICES LENGTH ---->", len(valid_services_map))

        only_deferral_activated = (len(valid_predispatch_services_map) == 1) and (valid_predispatch_services_map['Deferral'] is not None)
        no_other_services_activated = (len(valid_services_map) == 0)

        if only_deferral_activated and no_other_services_activated:
            deferral_only = True
            return deferral_only

        # If it hasn't already returned True at this point, then some services have been activated and the
        # deferral_only case is still False
        return deferral_only
