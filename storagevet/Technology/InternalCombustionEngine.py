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
Diesel

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.RotatingGenerator import RotatingGenerator
from storagevet.ErrorHandling import *


class ICE(RotatingGenerator):
    """ An Internal Combustion Engine

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        TellUser.debug(f"Initializing {__name__}")
        super().__init__(params)
        self.tag = 'ICE'
        self.efficiency = params['efficiency']  # gal/kWh
        self.fuel_cost = params['fuel_cost']    # $/gal, diesel

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        costs = super().objective_function(mask, annuity_scalar)
        total_out = self.variables_dict['elec'] + self.variables_dict['udis']
        # diesel fuel cost in $/kW
        costs[self.name + ' diesel_fuel_cost'] = cvx.sum(total_out * self.efficiency * self.fuel_cost * self.dt * annuity_scalar)

        return costs

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
        tech_id = self.unique_tech_id()
        fuel_col_name = tech_id + ' Diesel Fuel Costs'

        elec = self.variables_df['elec']
        analysis_years = self.variables_df.index.year.unique()
        fuel_costs_df = pd.DataFrame()
        for year in analysis_years:
            elec_sub = elec.loc[elec.index.year == year]
            # add diesel fuel costs in $/kW
            fuel_costs_df.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(self.efficiency * self.fuel_cost * self.dt * elec_sub)

        # fill forward
        fuel_costs_df = fill_forward_func(fuel_costs_df, None)
        # apply inflation rates
        fuel_costs_df = apply_inflation_rate_func(fuel_costs_df, None, min(analysis_years))
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, fuel_costs_df], axis=1)

        return pro_forma
