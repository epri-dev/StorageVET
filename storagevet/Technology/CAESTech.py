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
CAESTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from storagevet.Technology.EnergyStorage import EnergyStorage
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Library as Lib


class CAES(EnergyStorage):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        # create generic technology object
        super().__init__(params)
        # add CAES specific attributes
        self.tag = 'CAES'
        self.heat_rate_high = params['heat_rate_high']
        self.fuel_price = params['fuel_price']  # $/MillionBTU per month

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.fuel_price = Lib.fill_extra_data(self.fuel_price, years, 0, frequency)
        self.fuel_price = Lib.drop_extra_data(self.fuel_price, years)

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective costs for fuel cost and O&M cost

         Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            costs (Dict): Dict of objective costs

        """
        # get generic Tech objective costs
        costs = super().objective_function(mask, annuity_scalar)

        # add fuel cost expression
        fuel_exp = cvx.sum(cvx.multiply(self.fuel_price[mask].values, self.variables_dict['dis'] + self.variables_dict['udis'])
                           * self.heat_rate_high * self.dt * 1e-6 * annuity_scalar)
        costs.update({self.name + 'CAES_fuel_cost': fuel_exp})

        return costs

    def calc_operating_cost(self, energy_rate, fuel_rate):
        """ Calculates operating cost in dollars per MWh_out

         Args:
            energy_rate (float): energy rate [=] $/kWh
            fuel_rate (float): natural gas price rate [=] $/MillionBTU

        Returns:
            Value of Operating Cost [=] $/MWh_out

        Note: not used

        """
        fuel_cost = fuel_rate*self.heat_rate_high*1e3/1e6
        om = self.get_fixed_om()
        energy = energy_rate*1e3/self.rte

        return fuel_cost + om + energy

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = super().timeseries_report()
        results[tech_id + ' Natural Gas Price ($/MillionBTU)'] = self.fuel_price
        return results

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.


        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        fuel_col_name = self.unique_tech_id() + ' Natural Gas Costs'
        analysis_years = self.variables_df.index.year.unique()
        fuel_costs_df = pd.DataFrame()
        for year in analysis_years:
            fuel_price_sub = self.fuel_price.loc[self.fuel_price.index.year == year]
            fuel_costs_df.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(fuel_price_sub*self.heat_rate_high*1e3/1e6)
        # fill forward
        fuel_costs_df = fill_forward_func(fuel_costs_df, None)
        # apply inflation rates
        fuel_costs_df = apply_inflation_rate_func(fuel_costs_df, None, min(analysis_years))
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, fuel_costs_df], axis=1)
        return pro_forma
