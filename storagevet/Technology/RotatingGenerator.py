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
Rotating Generator Class

CT: (Combustion Turbine) or gas turbine
    - natural-gas fuel prices from monthly CSV
ICE (Internal Combustion Engine)
    - fuel_price fixed in Model Params CSV
DieselGenset or diesel engine-generator that is a single unit
    - can independently supply electricity allowing them to serve backup power
CHP (Combined Heat and Power)
    - natural-gas fuel prices from monthly CSV
    - also includes heat recovery

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER


class RotatingGenerator(DER):
    """ A Rotating Generator Technology

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """

        # create generic technology object
        super().__init__(params)
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.technology_type = 'Generator'
        self.rated_power = params['rated_capacity']  # kW/generator
        self.p_min = params['min_power']  # kW/generator
        self.variable_om = params['variable_om_cost']  # $/kwh
        self.fixed_om = params['fixed_om_cost']  # $/yr

        self.capital_cost_function = [params['ccost'],  # $/generator
                                      params['ccost_kW']]

        self.n = params['n']  # generators

        self.is_electric = True
        self.is_fuel = True

    def get_capex(self, **kwargs):
        """ Returns the capex of a given technology
                """
        return np.dot(self.capital_cost_function, [self.n, self.discharge_capacity()])

    def discharge_capacity(self):
        """

        Returns: the maximum discharge that can be attained

        """
        return self.rated_power * self.n

    def qualifying_capacity(self, event_length):
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        return self.discharge_capacity()

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            elec (Variable): A cvxpy variable equivalent to dis in batteries/CAES
                in terms of ability to provide services
            on (Variable): A cvxpy boolean variable for [...]

        Args:
            size (Int): Length of optimization variables to create

        """

        self.variables_dict = {'elec': cvx.Variable(shape=size, name=f'{self.name}-elecP', nonneg=True),
                               'udis': cvx.Variable(shape=size, name=f'{self.name}-udis', nonneg=True),
                               'on': cvx.Variable(shape=size, boolean=True, name=f'{self.name}-on')}

    def get_discharge(self, mask):
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return self.variables_dict['elec']

    def get_discharge_up_schedule(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.rated_power * self.n - self.variables_dict['elec'] - self.p_min * (1 - self.variables_dict['on'])

    def get_discharge_down_schedule(self, mask):
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['elec'] - self.p_min * self.variables_dict['on']

    def get_uenergy_decrease(self, mask):
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.dt * self.variables_dict['udis']

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        total_out = self.variables_dict['elec'] + self.variables_dict['udis']
        costs = {
            self.name + ' fixed': self.fixed_om * annuity_scalar,
            self.name + ' variable': cvx.sum(self.variable_om * self.dt * annuity_scalar * total_out),
        }
        return costs

    def constraints(self, mask):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        elec = self.variables_dict['elec']
        on = self.variables_dict['on']

        constraint_list += [cvx.NonPos((on * self.p_min) - elec)]
        constraint_list += [cvx.NonPos(elec - (on * self.rated_power * self.n))]

        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = pd.DataFrame(index=self.variables_df.index)
        solve_dispatch_opt = self.variables_df.get('elec')
        if solve_dispatch_opt is not None:
            results[tech_id + ' Electric Generation (kW)'] = \
                self.variables_df['elec']
            results[tech_id + ' On (y/n)'] = self.variables_df['on']
            results[tech_id + ' Energy Option (kWh)'] = \
                self.variables_df['udis'] * self.dt
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
        tech_id = self.unique_tech_id()
        if self.variables_df.empty:
            return pro_forma
        analysis_years = self.variables_df.index.year.unique()

        # OM COSTS
        om_costs = pd.DataFrame()
        variable_col_name = tech_id + ' Variable O&M Costs'
        elec = self.variables_df['elec']
        for year in analysis_years:
            elec_sub = elec.loc[elec.index.year == year]
            # add fixed o&m costs
            om_costs.loc[pd.Period(year=year, freq='y'), self.fixed_column_name()] = -self.fixed_om
            # add variable costs
            om_costs.loc[pd.Period(year=year, freq='y'), variable_col_name] = -np.sum(self.variable_om * self.dt * elec_sub)
        # fill forward
        om_costs = fill_forward_func(om_costs, None)
        # apply inflation rates
        om_costs = apply_inflation_rate_func(om_costs, None, min(analysis_years))
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        return pro_forma
