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
VoltVar.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
THIS CLASS HAS NOT BEEN VALIDATED OR TESTED TO SEE IF IT WOULD SOLVE.
"""

from storagevet.ValueStreams.ValueStream import ValueStream
import math
import pandas as pd
import logging
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib


class VoltVar(ValueStream):
    """ Reactive power support, voltage control, power quality. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
        """

        # generate the generic service object
        ValueStream.__init__(self, 'Volt Var', params)

        # add voltage support specific attributes
        self.vars_percent = params['percent'] / 100
        self.price = params['price']

        self.vars_reservation = 0

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.vars_percent = Lib.fill_extra_data(self.vars_percent, years, 0, 'M')
        self.vars_percent = Lib.drop_extra_data(self.vars_percent, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        # check to see if PV is included and is 'dc' connected to the  TODO fix this section if you want this service to work
        pv_max = 0
        inv_max = 0
        # if 'PV' in der_dict.keys:
        #     if der_dict['PV'].loc == 'dc':
        #         # use inv_max of the inverter shared by pv and ess and save pv generation
        #         inv_max = der_dict['PV'].inv_max
        #         pv_max = der_dict['PV'].generation
        # else:
        #     # otherwise just use the storage's rated discharge
        #     inv_max = der_dict['Storage'].dis_max_rated

        # # save load
        # self.load = load_data['load']

        self.vars_reservation = self.vars_percent * inv_max

        # constrain power s.t. enough vars are being outted as well
        power_sqrd = (inv_max**2) - (self.vars_reservation**2)

        dis_max = math.sqrt(power_sqrd) - pv_max
        ch_max = math.sqrt(power_sqrd)

        dis_min = 0
        ch_min = 0

        self.system_requirements = {
            'ch_max': ch_max,
            'dis_min': dis_min,
            'ch_min': ch_min,
            'dis_max': dis_max}

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
        proforma.columns = [self.name + ' Value']

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
