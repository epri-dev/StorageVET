"""
Copyright (c) 2023, Electric Power Research Institute

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
ValueStreamream.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
import numpy as np
import cvxpy as cvx
import pandas as pd


class ValueStream:
    """ A general template for services provided and constrained by the providing technologies.

    """

    def __init__(self, name, params):
        """ Initialize all services with the following attributes

        Args:
            name (str): A string name/description for the service
            params (Dict): input parameters
        """
        self.name = name
        self.dt = params['dt']
        self.system_requirements = []

        self.variables_df = pd.DataFrame()  # optimization variables are saved here
        self.variable_names = {}

        # attributes that are specific to the optimization problem being run (can change from window to window)
        self.variables = None

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        pass

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        pass

    def initialize_variables(self, size):
        """ Default method that will not create any new optimization variables_df

        Args:
            size (int): length of optimization variables_df to create

        """
    def p_reservation_charge_up(self, mask):
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def p_reservation_charge_down(self, mask):
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def p_reservation_discharge_up(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def p_reservation_discharge_down(self, mask):
        """ the amount of discharging power in the down direction (pulling power down from the grid) that
        needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def uenergy_option_stored(self, mask):
        """ the amount of energy, due to regulation up that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: the up energy reservation in kWh

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'ZeroStored{self.name}')

    def uenergy_option_provided(self, mask):
        """ the amount of energy, due to regulation up that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: the up energy reservation in kWh

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'ZeroProvided{self.name}')

    def worst_case_uenergy_stored(self, mask):
        """ the amount of energy, from the current SOE that needs to be reserved for this value stream
        to prevent any violates between the steps in time that are not catpured in our timeseries.

        Note: stored energy should be positive and provided energy should be negative

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: the case where the systems would end up with more energy than expected

        """
        stored = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'uEstoredZero{self.name}')
        return stored

    def worst_case_uenergy_provided(self, mask):
        """ the amount of energy, from the current SOE that needs to be reserved for this value stream
        to prevent any violates between the steps in time that are not catpured in our timeseries.

        Note: stored energy should be positive and provided energy should be negative

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: the case where the systems would end up with less energy than expected

        """
        provided = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'uEprovidedZero{self.name}')
        return provided

    def objective_function(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. [= charge - discharge]
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            A dictionary with the portion of  the objective function that it affects, labeled by the expression's key. Default is to return {}.
        """
        return {}

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. [= charge - discharge]
            combined_rating (cvx.Expression, int): the combined rating of DER that can reliabily dispatched in a worst case situtation
                these usually tend to be ESS and Generators

        Returns:
            An empty list (for aggregation of later constraints)
        """
        return []

    def save_variable_results(self, subs_index):
        """ Searches through the dictionary of optimization variables_df and saves the ones specific to each
        ValueStream instance and saves the values it to itself

        Args:
            subs_index (Index): index of the subset of data for which the variables_df were solved for

        """
        variable_values = pd.DataFrame({name: self.variables[name].value for name in self.variable_names}, index=subs_index)
        self.variables_df = pd.concat([self.variables_df, variable_values], sort=True)

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """

    def monthly_report(self):
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, **kwargs):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        return {}

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        pass

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
        opt_years = [pd.Period(year=item, freq='y') for item in opt_years]
        proforma = pd.DataFrame(index=opt_years)
        return proforma

    def min_regulation_up(self):
        return 0

    def min_regulation_down(self):
        return 0

    def max_participation_is_defined(self):
        return False

    def rte_list(self, poi):
        # value streams sometimes need rte in calculations
        # get a list of rte values from all active ess
        # default to [1], so that division by rte remains valid
        rte_list = [der.rte for der in poi.der_list if der.technology_type == 'Energy Storage System']
        if len(rte_list) == 0:
            rte_list = [1]
        # set an attribute to the value stream
        self.rte_list = rte_list
