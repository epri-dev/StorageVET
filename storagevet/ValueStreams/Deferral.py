"""
Copyright (c) 2021, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
Deferral.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
import numpy as np
import cvxpy as cvx
from storagevet.ValueStreams.ValueStream import ValueStream
import pandas as pd
import storagevet.Library as Lib
import random
from storagevet.ErrorHandling import *
from storagevet.Library import truncate_float


class Deferral(ValueStream):
    """ Investment deferral. Each service will be daughters of the PreDispService class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
        """

        # generate the generic service object
        ValueStream.__init__(self, 'Deferral', params)

        # add Deferral specific attributes
        self.max_import = params['planned_load_limit']  # positive
        self.max_export = params['reverse_power_flow_limit']  # negative
        self.last_year = params['last_year'].year
        self.year_failed = params['last_year'].year + 1
        self.min_years = params.get('min_year_objective', 0)
        self.load = params['load']  # deferral load
        self.growth = params['growth']  # Growth Rate of deferral load (%/yr)
        self.price = params['price']  # $/yr

        self.p_min = 0
        self.e_min = 0
        self.deferral_df = None
        self.e_walk = pd.Series()
        self.power_requirement = pd.Series()

    def check_for_deferral_failure(self, end_year, poi, frequency, opt_years, def_load_growth):
        """This functions checks the constraints of the storage system against any predispatch or user inputted constraints
        for any infeasible constraints on the system.

        The goal of this function is to predict the year that storage will fail to deferral a T&D asset upgrade.

        Only runs if Deferral is active.

        Args:
            end_year:
            poi:
            frequency:
            opt_years:
            def_load_growth:

        Returns: new list of optimziation years

        """
        TellUser.info('Finding first year of deferral failure...')
        current_year = self.load.index.year[-1]

        additional_years = [current_year]
        try:
            find_failure_year = not poi.is_sizing_optimization
        except AttributeError:
            find_failure_year = True

        # get list of RTEs
        rte_lst = [der.rte for der in poi.der_list if der.technology_type == 'Energy Storage System']
        ess_cha_max = 0
        ess_dis_max = 0
        ess_ene_max = 0
        conventional_gen_max = 0
        for der_isnt in poi.der_list:
            if der_isnt.technology_type == "Energy Storage System":
                ess_cha_max += der_isnt.ch_max_rated
                ess_dis_max += der_isnt.dis_max_rated
                ess_ene_max += der_isnt.ene_max_rated * der_isnt.ulsoc
            if der_isnt.technology_type == 'Generator':
                conventional_gen_max += der_isnt.discharge_capacity()
        years_deferral_column = []
        min_power_deferral_column = []
        min_energy_deferral_column = []

        while current_year <= end_year.year:
            size = len(self.load)
            years_deferral_column.append(current_year)

            # TODO can we check the max year? or can we take the previous answer to calculate the next energy requirements?
            positive_feeder_load = self.load.values
            negative_feeder_load = np.zeros(size)
            for der_isnt in poi.der_list:
                if der_isnt.technology_type == "Load":
                    positive_feeder_load = positive_feeder_load + der_isnt.value.values
                if der_isnt.technology_type == "Intermittent Resource" and not der_isnt.being_sized():
                    # TODO: should take PV variability into account here
                    negative_feeder_load = negative_feeder_load - der_isnt.maximum_generation()

            positive_feeder_load += np.repeat(conventional_gen_max, size)
            # Determine power requirement of the storage:
            # (1) anytime the net_feeder_load goes above deferral_max_import (too much load)
            positive_load_power_req = positive_feeder_load - self.max_import
            positive_power_req = positive_load_power_req.clip(min=0)
            # (2) anytime the net_feeder_load goes below deferral_max_exports
            # (assumes deferral_max_export < 0)  (too much generation)
            negative_load_power_req = negative_feeder_load - self.max_export
            negative_power_req = negative_load_power_req.clip(max=0)
            # The sum of (1) and (2)
            storage_power_requirement = positive_power_req + negative_power_req

            e_walk, _ = self.precheck_failure(self.dt, rte_lst, storage_power_requirement)
            TellUser.debug(f'In {current_year} -- min power: {truncate_float(self.p_min)}  min energy: {truncate_float(self.e_min)}')
            # save min power and energy requirements
            min_power_deferral_column.append(self.p_min)
            min_energy_deferral_column.append(self.e_min)
            # save energy required as function of time & storage power required as function of time
            self.e_walk = pd.Series(e_walk, index=self.load.index)
            self.power_requirement = pd.Series(storage_power_requirement, index=self.load.index)
            if find_failure_year and (self.p_min > ess_dis_max or self.p_min > ess_cha_max or self.e_min > ess_ene_max):
                # then we predict that deferral will fail
                last_deferral_yr = current_year - 1
                self.set_last_deferral_year(last_deferral_yr, current_year)

                opt_years = list(set(opt_years + additional_years))
                find_failure_year = False
                TellUser.info(f'{self.name} updating analysis years: {opt_years}')

            # the current year we have could be the last year the deferral is possible, so we want
            # to keep it in self.opt_results until we know the next is can be deferred as well
            additional_years = [current_year, current_year + 1]
            next_opt_years = list(set(opt_years + additional_years))

            # add additional year of data to der data
            for der in poi.der_list:
                der.grow_drop_data(next_opt_years, frequency, def_load_growth)

            # add additional year of data to deferred load
            self.grow_drop_data(next_opt_years, frequency, def_load_growth)

            # index the current year by one
            current_year += 1

        self.deferral_df = pd.DataFrame({'Year': years_deferral_column,
                                         'Power Capacity Requirement (kW)': min_power_deferral_column,
                                         'Energy Capacity Requirement (kWh)': min_energy_deferral_column})
        self.deferral_df.set_index('Year', inplace=True)
        return opt_years

    def precheck_failure(self, tstep, rte_lst, sto_p_req):
        """
        This function takes in a vector of storage power requirements (negative=charging and positive=discharging) [=] kW
        that are required to perform the deferral as well as a time step (tstep) [=] hrs

        Args:
            tstep (float): timestep of the data in hours
            rte_lst (list): round trip efficiency of storage
            sto_p_req (list, ndarray): storage power requirement

        Returns:
            how much the energy in the ESS needs to wander as a function of time,
            theoretical dispatch of the ESS to meet on feeder limits

        Notes:
            This algorithm can reliably find the last year deferral is possible, however the problem might still
            be found INFEASIBLE if the ESS cannot use it's full range of SOC (ie. if LLSOC is too high or ULSOC is too low)
        """
        # Loop through time steps. If the storage is forced to dispatch from the constraint,
        # return to nominal SOC as soon as possible after.
        self.p_min = max(abs(sto_p_req))
        # TODO: determine min energy requirement in static recursive function to speed runtime --HN
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
                random_rte = random.choice(rte_lst)
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep * random_rte
            elif e_walk[step - 1] < 0:  # Otherwise contribute its full power to returning energy to nominal
                sto_dispatch[step] = -min(abs(self.p_min), abs(e_walk[step - 1] / tstep), abs(self.max_import - self.load.iloc[step]))
                random_rte = random.choice(rte_lst)
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep * random_rte  # kWh
            elif e_walk[step - 1] > 0:
                sto_dispatch[step] = min(abs(self.p_min), abs(e_walk[step - 1] / tstep))
                e_walk[step] = e_walk[step - 1] - sto_dispatch[step] * tstep  # kWh
            else:
                sto_dispatch[step] = 0
                e_walk[step] = e_walk[step - 1]
        kwh_min = max(e_walk) - min(e_walk)
        self.e_min = float(kwh_min)
        return e_walk, sto_dispatch

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.load = Lib.fill_extra_data(self.load, years, self.growth, frequency)
        self.load = Lib.drop_extra_data(self.load, years)

    def set_last_deferral_year(self, last_year, failed_year):
        """Sets last year that deferral is possible

        Args:
            last_year (int): The last year storage can defer an T&D equipment upgrade
            failed_year (int): the year that deferring an upgrade will fail
        """
        self.last_year = last_year
        self.year_failed = failed_year
        TellUser.info(f'{self.name} year failed set to: ' + str(self.year_failed))

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. [= charge - discharge]
            combined_rating (Dictionary): the combined rating of each DER class type

        Returns:
            An empty list (for aggregation of later constraints)
        """
        # adding constraints to ensure power dispatch does not violate thermal limits of transformer deferred
        # only include them if deferral is not going to fail
        constraints = []
        year_of_optimization = mask.loc[mask].index.year[-1]
        if year_of_optimization < self.year_failed:
            load_beyond_poi = cvx.Parameter(value=self.load.loc[mask].values, name='deferral_load', shape=sum(mask))
            # -(max export) >= dis - ch + generation - loads
            constraints += [cvx.NonPos(self.max_export - load_sum - load_beyond_poi + (-1)*net_ess_power + generator_out_sum + tot_variable_gen)]
            # max import >= loads - (dis - ch) - generation
            constraints += [cvx.NonPos(load_sum + load_beyond_poi + net_ess_power + (-1)*generator_out_sum + (-1)*tot_variable_gen - self.max_import)]
            # TODO make sure power does doesn't violate the constraints during dispatch service activity
        else:
            TellUser.debug(f"{self.name} did not add any constraints to our system of equations")

        return constraints

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.load.index)
        report.loc[:, 'Deferral: Load (kW)'] = self.load
        report.loc[:, 'Deferral: Energy Requirement (kWh)'] = -self.e_walk
        report.loc[:, 'Deferral: Power Requirement (kW)'] = self.power_requirement
        return report.sort_index()

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame): DataFrame with all the optimization variable solutions

        Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided)

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
        # apply inflation rates
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))

        return proforma

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, **kwargs):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        return {'deferral_results': self.deferral_df}
