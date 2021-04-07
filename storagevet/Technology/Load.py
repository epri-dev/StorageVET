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
Load

This Python class contains methods and attributes specific for technology analysis within DERVET.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.0.2'

import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER
import storagevet.Library as Lib
import cvxpy as cvx
from storagevet.ErrorHandling import *


class Load(DER):
    """ An Site Load object

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        super().__init__(params)
        self.technology_type = 'Load'
        self.tag = 'Load'
        self.dt = params['dt']
        self.value = params['site_load']

    def zero_column_name(self):
        return None  # Loads do not have capital costs

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.value = Lib.fill_extra_data(self.value, years, load_growth, frequency)
        self.value = Lib.drop_extra_data(self.value, years)

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return cvx.Parameter(value=self.value[mask].values, shape=sum(mask), name='SiteLoad')

    def effective_load(self):
        """ Returns the load that is seen by the microgrid or point of interconnection

        """
        return self.value.loc[:]

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables_df.index)
        results[f"{self.unique_tech_id()} Original Load (kW)"] = \
            self.value.loc[:]
        return results

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        # DESIGN PLOT (peak load day)
        max_day = time_series_data['Total Load (kW)'].idxmax().date()
        max_day_data = time_series_data[time_series_data.index.date == max_day]
        time_step = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        peak_day_load = pd.DataFrame({'Date': max_day_data.index.date,
                                      'Load (kW)': max_day_data['Total Load (kW)'].values,
                                      'Net Load (kW)': max_day_data['Net Load (kW)'].values}, index=time_step)
        return {'peak_day_load': peak_day_load}
