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
DAEnergyTimeShift.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import cvxpy as cvx
import pandas as pd
import storagevet.Library as Lib
import numpy as np


class DAEnergyTimeShift(ValueStream):
    """ Day-Ahead Energy Time Shift. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
        """
        ValueStream.__init__(self, 'DA', params)
        self.price = params['price']
        self.growth = params['growth']  # growth rate of energy prices (%/yr)

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation


        """
        self.price = Lib.fill_extra_data(self.price, years, self.growth, frequency)
        self.price = Lib.drop_extra_data(self.price, years)

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
            A dictionary with the portion of the objective function that it affects, labeled by the expression's key. Default is to return {}.
        """
        p_da = cvx.Parameter(value=self.price.loc[mask].values, shape=sum(mask), name='DA_price')
        cost = cvx.sum(cvx.multiply(p_da, net_ess_power) + cvx.multiply(-p_da, generator_out_sum) + cvx.multiply(-p_da, tot_variable_gen) + cvx.multiply(p_da, load_sum)) 
        return {self.name: cost * annuity_scalar * self.dt}

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, 'Energy Price ($/kWh)'] = self.price
        return report

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, **kwargs):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        df_dict = dict()
        energy_price = time_series_data.loc[:, 'Energy Price ($/kWh)'].to_frame()
        energy_price.loc[:, 'date'] = time_series_data.index.date
        energy_price.loc[:, 'hour'] = (time_series_data.index + pd.Timedelta('1s')).hour + 1  # hour ending
        energy_price = energy_price.reset_index(drop=True)
        df_dict['energyp_map'] = energy_price.pivot_table(values='Energy Price ($/kWh)', index='hour', columns='date')
        return df_dict

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame): DataFrame with all the optimization variable solutions

        Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided), a list (of columns that remain zero), and a list (of columns that
        retain a constant value over the entire project horizon).
        """
        proforma = super().proforma_report(opt_years, apply_inflation_rate_func,
                                           fill_forward_func, results)
        energy_cost = self.dt * np.multiply(results['Total Generation (kW)'] + results['Total Storage Power (kW)'], self.price)
        for year in opt_years:
            year_subset = energy_cost[energy_cost.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), 'DA ETS'] = year_subset.sum()
        proforma = fill_forward_func(proforma, self.growth)
        return proforma

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
        except KeyError:
            pass
