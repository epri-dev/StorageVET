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
CAESTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from storagevet.Technology.EnergyStorage import EnergyStorage
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class CAES(EnergyStorage):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        TellUser.debug(f"Initializing {__name__}")
        self.tag = 'CAES'
        # create generic technology object
        super().__init__(params)
        # add CAES specific attributes
        self.tag = 'CAES'
        self.fuel_type = params['fuel_type']
        self.heat_rate = 1e-3 * params['heat_rate_high']   # MMBtu/MWh ---> MMBtu/kWh
        self.is_fuel = True

    def initialize_degradation_module(self, opt_agg):
        """
        bypass degradation by doing nothing here
        """
        pass

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
        total_out = self.variables_dict['dis'] + self.variables_dict['udis']
        # add fuel cost expression in $/kWh
        fuel_exp = cvx.sum(total_out * self.heat_rate * self.fuel_cost * self.dt * annuity_scalar)
        costs.update({self.name + ' fuel_cost': fuel_exp})

        return costs

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        if self.variables_df.index.empty:
            return pro_forma
        tech_id = self.unique_tech_id()
        optimization_years = self.variables_df.index.year.unique()
        dis = self.variables_df['dis']
        udis = self.variables_df['udis']

        # add CAES fuel costs in $/kW
        fuel_costs = pd.DataFrame()
        fuel_col_name = tech_id + ' Fuel Costs'
        for year in optimization_years:
            dis_sub = dis.loc[dis.index.year == year]
            udis_sub = udis.loc[udis.index.year == year]
            # add fuel costs in $/kW
            fuel_costs.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(self.heat_rate * self.fuel_cost * self.dt * (dis_sub + udis_sub))
        # fill forward
        fuel_costs = fill_forward_func(fuel_costs, None)
        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, fuel_costs], axis=1)

        return pro_forma
