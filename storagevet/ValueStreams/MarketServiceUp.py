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
MarketServiceUp.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import numpy as np
import cvxpy as cvx
import pandas as pd
import storagevet.Library as Lib


class MarketServiceUp(ValueStream):
    """ A market service that provides service only through bringing demand down

    """

    def __init__(self, name, full_name, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            name (str): abbreviated name
            full_name (str): the expanded name of the service
            params (Dict): input parameters
        """
        ValueStream.__init__(self, name, params)
        self.price = params['price']
        self.growth = params['growth']/100  # growth rate of spinning reserve price (%/yr)
        self.duration = params['duration']
        self.full_name = full_name
        self.variable_names = {'ch_less', 'dis_more'}
        self.variables_df = pd.DataFrame(columns=self.variable_names)

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be
        called after add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this
                simulation

        """
        self.price = Lib.fill_extra_data(self.price, years, self.growth, frequency)
        self.price = Lib.drop_extra_data(self.price, years)

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            dis_more (Variable): A cvxpy variable for spinning reserve capacity to increase
                discharging power
            ch_less (Variable): A cvxpy variable for spinning reserve capacity to decrease
                charging power

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        self.variables = {'ch_less': cvx.Variable(shape=size, name=f'{self.name}_ch_less'),
                          'dis_more': cvx.Variable(shape=size, name=f'{self.name}_dis_more')}

    def objective_function(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                           net_ess_power, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the
                system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the
                system. [= charge - discharge]
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit
                that helps capture the cost/benefit over the entire project lifetime (only to be
                set iff sizing)

        Returns:
            The expression of the objective function that it affects. This can be passed into the
            cvxpy solver.

        """
        payment = cvx.Parameter(sum(mask), value=self.price.loc[mask].values,
                                name=f'{self.name}_price')

        return {
            self.name: cvx.sum(
                cvx.multiply(-payment, self.variables['ch_less']) +
                cvx.multiply(-payment, self.variables['dis_more'])) * self.dt * annuity_scalar}

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power,
                    combined_rating):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within
                the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the
                system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class type

        Returns:
            constraint_list (list): list of constraints

        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-self.variables['ch_less'])]
        constraint_list += [cvx.NonPos(-self.variables['dis_more'])]
        return constraint_list

    def p_reservation_charge_up(self, mask):
        """ the amount of charging power in the up direction (supplying power up into the grid)
        that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['ch_less']

    def p_reservation_discharge_up(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid)
        that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['dis_more']

    def worst_case_uenergy_provided(self, mask):
        """ the amount of energy, from the current SOE that needs to be reserved for this value
        stream to prevent any violates between the steps in time that are not catpured in our
        timeseries.

        Note: stored energy should be positive and provided energy should be negative

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: tuple (stored, provided),
            where the first value is the case where the systems would end up with more energy than
            expected and the second corresponds to the case where the systems would end up with
            less energy than expected

        """
        provided = self.variables['ch_less']*self.duration + self.variables['dis_more']*self.duration
        return provided

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the
            results pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, f"{self.name} Price ($/kW)"] = self.price
        report.loc[:, f"{self.full_name} Up (Charging) (kW)"] = self.variables_df['ch_less']
        report.loc[:, f"{self.full_name} Up (Discharging) (kW)"] = self.variables_df['dis_more']

        return report

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided)
        """
        proforma = super().proforma_report(opt_years, apply_inflation_rate_func,
                                           fill_forward_func, results)
        bid = \
            results.loc[:, f'{self.full_name} Up (Charging) (kW)'] + \
            results.loc[:, f'{self.full_name} Up (Discharging) (kW)']
        spinning_prof = np.multiply(bid, self.price) * self.dt

        for year in opt_years:
            year_subset = spinning_prof[spinning_prof.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), self.full_name] = year_subset.sum()
        # forward fill growth columns with inflation at growth rate
        proforma = fill_forward_func(proforma, self.growth)

        return proforma

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not
        require all price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = time_series_data.loc[:, f'{self.name} Price ($/kW)']
        except KeyError:
            pass
