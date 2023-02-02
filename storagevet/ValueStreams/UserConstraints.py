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
UserConstraints.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import pandas as pd
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib
from storagevet.ErrorHandling import *
import numpy as np

VERY_LARGE_NUMBER = 2**32 - 1
VERY_LARGE_NEGATIVE_NUMBER = -1 * VERY_LARGE_NUMBER

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
        self.poi_import_min_constraint = None
        self.poi_import_max_constraint = None
        self.poi_export_min_constraint = None
        self.poi_export_max_constraint = None
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
        # NOTE: because of this, we handle zeroes in min constraints here (we substitute in very large negative values)
        #       do not change any max constraints (those zeroes are important to control any no-export or no-import cases)
        self.poi_export_max_constraint = self.user_power.get('POI: Max Export (kW)')
        if self.poi_export_max_constraint is not None:
            self.poi_export_max_constraint = self.return_positive_values(self.poi_export_max_constraint)
            self.system_requirements.append(Requirement('poi export', 'max', self.name, self.poi_export_max_constraint))

        self.poi_export_min_constraint = self.user_power.get('POI: Min Export (kW)')
        if self.poi_export_min_constraint is not None:
            self.poi_export_min_constraint = self.return_positive_values(self.poi_export_min_constraint)
            self.poi_export_min_constraint[self.poi_export_min_constraint == 0] = VERY_LARGE_NEGATIVE_NUMBER
            TellUser.info('In order for the POI: Min Export constraint to work, we modify values that are zero to be a very large negative number')
            self.system_requirements.append(Requirement('poi export', 'min', self.name, self.poi_export_min_constraint))

        self.poi_import_max_constraint = self.user_power.get('POI: Max Import (kW)')
        if self.poi_import_max_constraint is not None:
            self.poi_import_max_constraint = self.return_positive_values(self.poi_import_max_constraint)
            self.system_requirements.append(Requirement('poi import', 'max', self.name, self.poi_import_max_constraint))

        self.poi_import_min_constraint = self.user_power.get('POI: Min Import (kW)')
        if self.poi_import_min_constraint is not None:
            self.poi_import_min_constraint = self.return_positive_values(self.poi_import_min_constraint)
            self.poi_import_min_constraint[self.poi_import_min_constraint == 0] = VERY_LARGE_NEGATIVE_NUMBER
            TellUser.info('In order for the POI: Min Import constraint to work, we modify values that are zero to be a very large negative number')
            self.system_requirements.append(Requirement('poi import', 'min', self.name, self.poi_import_min_constraint))

        # set system requirements on Energy
        self.soe_max_constraint = self.user_energy.get('Aggregate Energy Max (kWh)')
        if self.soe_max_constraint is not None:
            self.system_requirements.append(Requirement('energy', 'max', self.name, self.soe_max_constraint))
        self.soe_min_constraint = self.user_energy.get('Aggregate Energy Min (kWh)')
        if self.soe_min_constraint is not None:
            self.system_requirements.append(Requirement('energy', 'min', self.name, self.soe_min_constraint))

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        # use the altered system requirement constraints in the output time series
        # NOTE: we display export values as the negative of what they are in the constraint,
        #     since negative Net Power is actually positive Export Power
        if self.poi_export_max_constraint is not None:
            TellUser.info('For better alignment with "Net Power" in the output time series, we multiply POI: Max Export values by -1')
            self.user_power['POI: Max Export (kW)'] = self.poi_export_max_constraint * -1
        if self.poi_export_min_constraint is not None:
            TellUser.info('For better alignment with "Net Power" in the output time series, we multiply POI: Min Export values by -1')
            self.user_power['POI: Min Export (kW)'] = self.poi_export_min_constraint * -1
        if self.poi_import_max_constraint is not None:
            self.user_power['POI: Max Import (kW)'] = self.poi_import_max_constraint
        if self.poi_import_min_constraint is not None:
            self.user_power['POI: Min Import (kW)'] = self.poi_import_min_constraint
        # add 'User Constraints' label to beginning of each column name
        new_power_names = {original: f"{self.name} {original}" for original in self.user_power.columns}
        self.user_power.rename(columns=new_power_names, inplace=True)
        new_energy_names = {original: f"{self.name} {original}" for original in self.user_energy.columns}
        self.user_energy.rename(columns=new_energy_names, inplace=True)
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
        proforma = fill_forward_func(proforma, None)
        return proforma

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value

    @staticmethod
    def return_positive_values(array):
        """ Given an array s.t. for all values >0 or for all values <0 is true,
        return an array whose values are always positive

        Args:
            array (pd.Series):

        Returns: Series, with its values changed

        """
        if (array > 0).any():
            return array
        else:
            return -1*array
