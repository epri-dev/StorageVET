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
UserConstraints.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import pandas as pd
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib
import numpy as np


class UserConstraints(ValueStream):
    """ User entered time series constraints. Each service will be daughters of the PreDispService class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Acceptable constraint names are: 'Power Max (kW)', 'Power Min (kW)', 'Energy Max (kWh)', 'Energy Min (kWh)'

          Args:
            params (Dict): input parameters
        """
        # generate the generic service object
        ValueStream.__init__(self, 'User Constraints', params)

        self.user_power = params['power']
        self.user_energy = params['energy']
        self.price = params['price']  # $/yr
        self.charge_min_constraint = None
        self.charge_max_constraint = None
        self.discharge_min_constraint = None
        self.discharge_max_constraint = None
        self.soe_min_constraint = None
        self.soe_max_constraint = None

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.user_power = Lib.fill_extra_data(self.user_power, years, 0, frequency)
        self.user_power = Lib.drop_extra_data(self.user_power, years)

        self.user_energy = Lib.fill_extra_data(self.user_energy, years, 0, frequency)
        self.user_energy = Lib.drop_extra_data(self.user_energy, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        # set system requirements on power (make sure everything is positive, regardless of how user gave the values)
        self.discharge_max_constraint = self.user_power.get('POI: Max Export (kW)')
        if self.discharge_max_constraint is not None:
            self.discharge_max_constraint = self.return_positive_values(self.discharge_max_constraint)
            self.system_requirements.append(Requirement('discharge', 'max', self.name, self.discharge_max_constraint))
        self.discharge_min_constraint = self.user_power.get('POI: Min Export (kW)')
        if self.discharge_min_constraint is not None:
            self.discharge_min_constraint = self.return_positive_values(self.discharge_min_constraint)
            self.system_requirements.append(Requirement('discharge', 'min', self.name, self.discharge_min_constraint))
        self.check_if_feasible(self.discharge_max_constraint, self.discharge_min_constraint)
        self.charge_max_constraint = self.user_power.get('POI: Max Import (kW)')
        if self.charge_max_constraint is not None:
            self.charge_max_constraint = self.return_positive_values(self.charge_max_constraint)
            self.system_requirements.append(Requirement('charge', 'max', self.name, self.charge_max_constraint))
        self.charge_min_constraint = self.user_power.get('POI: Min Import (kW)')
        if self.charge_min_constraint is not None:
            self.charge_min_constraint = self.return_positive_values(self.charge_min_constraint)
            self.system_requirements.append(Requirement('charge', 'min', self.name, self.charge_min_constraint))
        self.check_if_feasible(self.charge_max_constraint, self.charge_min_constraint)
        if self.charge_min_constraint is not None and self.discharge_min_constraint is not None:
            # both cannot be nonzero at the same time
            if (self.charge_min_constraint > 0 and self.discharge_min_constraint > 0).any():
                raise Exception('User given constraints are not feasible.')

        # set system requirements on Energy
        self.soe_max_constraint = self.user_energy.get('Aggregate Energy Max (kWh)')
        if self.soe_max_constraint is not None:
            self.system_requirements.append(Requirement('energy', 'max', self.name, self.soe_max_constraint))
        self.soe_min_constraint = self.user_energy.get('Aggregate Energy Min (kWh)')
        if self.soe_min_constraint is not None:
            self.system_requirements.append(Requirement('energy', 'min', self.name, self.soe_min_constraint))
        self.check_if_feasible(self.soe_max_constraint, self.soe_min_constraint)

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        # add 'User-" to beginging of each column name
        new_power_names = {original: f"{self.name}-{original}" for original in self.user_power.columns}
        self.user_power.rename(columns=new_power_names)
        new_energy_names = {original: f"{self.name}-{original}" for original in self.user_energy.columns}
        self.user_energy.rename(columns=new_energy_names)
        # concat energy and power together
        power_df = self.user_power if not self.user_power.empty else None
        energy_df = self.user_energy if not self.user_energy.empty else None
        report = pd.concat([power_df, energy_df], axis=1)
        return report

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
        proforma = ValueStream.proforma_report(self, opt_years, apply_inflation_rate_func,
                                               fill_forward_func, results)
        proforma[self.name + ' Value'] = 0

        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.price
        # apply inflation rates
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))
        return proforma

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value

    @staticmethod
    def check_if_feasible(max_array, min_array):
        """Checks if the two arrays will create a feasible constraint. If they do not, then an Exception
        is raised.

        Args:
            max_array (pd.Series): the array that will constrain a value to be smaller than
            min_array (pd.Series): The array that will constrain a value to be bigger than
        """
        # if both a max and min are not empty, then preform this check to check for infeasibility
        if (max_array is not None or min_array is not None) and (min_array > max_array).any():
            raise Exception('User given constraints are not feasible.')

    @staticmethod
    def return_positive_values(array):
        """ Given an array s.t. for all values >0 or for all values <0 is true,
        return an array whose values are always positive

        Args:
            array (pd.Series):

        Returns: Series, with its values changed

        """
        if (array >= 0).any():
            return array
        else:
            return -1*array
