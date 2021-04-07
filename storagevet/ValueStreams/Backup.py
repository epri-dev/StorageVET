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
Backup.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib
import pandas as pd
import numpy as np


class Backup(ValueStream):
    """ Backup Power Service. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
        """

        # generate the generic service object
        ValueStream.__init__(self, 'Backup', params)
        self.energy_req = params['daily_energy']
        self.monthly_energy = params['monthly_energy']  # raw input form of energy requirement
        self.price = params['monthly_price']

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.energy_req = Lib.fill_extra_data(self.energy_req, years, 0, frequency)
        self.energy_req = Lib.drop_extra_data(self.energy_req, years)
        self.monthly_energy = Lib.fill_extra_data(self.monthly_energy, years, 0, 'M')
        self.monthly_energy = Lib.drop_extra_data(self.monthly_energy, years)
        self.price = Lib.fill_extra_data(self.price, years, 0, 'M')
        self.price = Lib.drop_extra_data(self.price, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        # backup energy adds a minimum energy level
        self.system_requirements.append(Requirement('SOE', 'min', self.name, self.energy_req))

    def monthly_report(self):
        """  Calculates the monthly cost or benefit of the service and adds them to the monthly financial result dataframe

        Returns: A dataframe with the monthly input price of the service and the calculated monthly value in respect
                for each month

        """

        monthly_financial_result = pd.DataFrame({'Backup Price ($/kWh)': self.price}, index=self.price.index)
        monthly_financial_result.index.names = ['Year-Month']

        return monthly_financial_result

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
        proforma[self.name] = 0

        for year in opt_years:
            monthly_benefit = np.multiply(self.monthly_energy, self.price)
            proforma.loc[pd.Period(year=year, freq='y')] = monthly_benefit.sum()
        # apply inflation rates
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))

        return proforma

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.energy_req.index)
        report.loc[:, 'Backup Energy Reserved (kWh)'] = self.energy_req
        return report

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = monthly_data.loc[:, 'Backup Price ($/kWh)']
        except KeyError:
            pass
