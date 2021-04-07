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
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from storagevet.Technology.DistributedEnergyResource import DER
import cvxpy as cvx
import pandas as pd
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class PV(DER):
    """ Assumes perfect foresight. Ability to curtail PV generation

    """

    def __init__(self, params):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        super().__init__(params)
        self.technology_type = "Intermittent Resource"
        self.tag = 'PV'
        self.growth = params['growth']
        self.curtail = True
        self.gen_per_rated = params['rated gen']
        self.rated_capacity = params['rated_capacity']
        self.loc = params['loc'].lower()
        self.grid_charge = params['grid_charge']
        self.inv_max = params['inv_max']
        self.capital_cost_function = params['ccost_kW']
        self.fixed_om = params['fixed_om_cost']  # $/yr

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.gen_per_rated = Lib.fill_extra_data(self.gen_per_rated, years, self.growth, frequency)
        self.gen_per_rated = Lib.drop_extra_data(self.gen_per_rated, years)

    def get_capex(self, **kwargs):
        """

        Returns: the capex of this DER for optimization

        """
        return self.capital_cost_function * self.rated_capacity

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create

        """
        if self.curtail:
            self.variables_dict = {'pv_out': cvx.Variable(shape=size, name='pv_out', nonneg=True)}

    def maximum_generation(self, label_selection=None, **kwargs):
        """ The most that the PV system could discharge.

        Args:
            label_selection: A single label, e.g. 5 or 'a',
                a list or array of labels, e.g. ['a', 'b', 'c'],
                a boolean array of the same length as the axis being sliced, e.g. [True, False, True]
                a callable function with one argument (the calling Series or DataFrame)

        Returns: valid array output for indexing (one of the above) of the max generation profile

        """
        if label_selection is None:
            return self.gen_per_rated.values * self.rated_capacity

        else:
            return self.gen_per_rated.loc[label_selection].values * self.rated_capacity

    def get_discharge(self, mask):
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        if self.curtail:
            return self.variables_dict['pv_out']
        else:
            return cvx.Parameter(shape=sum(mask), name='pv_out', value=self.maximum_generation(mask))

    def constraints(self, mask, **kwargs):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []

        if self.loc == 'ac':
            constraint_list += [cvx.NonPos(self.get_discharge(mask) - self.inv_max)]
            constraint_list += [cvx.NonPos(- self.inv_max - self.get_discharge(mask))]
        if self.curtail:
            constraint_list += [cvx.NonPos(self.get_discharge(mask) - self.maximum_generation(mask))]

        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = pd.DataFrame(index=self.gen_per_rated.index)
        if self.curtail:
            solve_dispatch_opt = self.variables_df.get('pv_out')
            if solve_dispatch_opt is not None:
                results[tech_id + ' Electric Generation (kW)'] = \
                    self.variables_df['pv_out']
        else:
            results[tech_id + ' Electric Generation (kW)'] = self.maximum_generation()
        results[tech_id + ' Maximum (kW)'] = self.maximum_generation()
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
        if self.variables_df.empty:
            return pro_forma
        analysis_years = self.variables_df.index.year.unique()

        # OM COSTS
        om_costs = pd.DataFrame()
        for year in analysis_years:
            # add fixed o&m costs
            om_costs.loc[pd.Period(year=year, freq='y'), self.fixed_column_name()] = -self.fixed_om
        # fill forward
        om_costs = fill_forward_func(om_costs, None)
        # apply inflation rates
        om_costs = apply_inflation_rate_func(om_costs, None, min(analysis_years))
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        return pro_forma
