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
LoadFollowing.py

This Python class contains methods and attributes specific for service analysis
within StorageVet.
"""
from storagevet.ValueStreams.MarketServiceUpAndDown import MarketServiceUpAndDown
import cvxpy as cvx
import numpy as np
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class LoadFollowing(MarketServiceUpAndDown):
    """ Load Following.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
        """
        MarketServiceUpAndDown.__init__(self, 'LF', 'Load Following', params)
        self.u_ts_constraints = params.get('u_ts_constraints', False)
        self.d_ts_constraints = params.get('d_ts_constraints', False)
        if self.u_ts_constraints:
            self.regu_max = params['lf_u_max']
            self.regu_min = params['lf_u_min']
        if self.d_ts_constraints:
            self.regd_max = params['lf_d_max']
            self.regd_min = params['lf_d_min']

        if self.dt > 0.25:
            TellUser.warning("WARNING: using Load Following Service and " +
                             "time series timestep is greater than 15 min.")

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that
        might have slipped in. Update variable that hold timeseries data
        after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of
                loads in this simulation

        """
        super().grow_drop_data(years, frequency, load_growth)
        self.eou_avg = Lib.fill_extra_data(self.eou_avg, years, 0, frequency)
        self.eou_avg = Lib.drop_extra_data(self.eou_avg, years)

        self.eod_avg = Lib.fill_extra_data(self.eod_avg, years, 0, frequency)
        self.eod_avg = Lib.drop_extra_data(self.eod_avg, years)

        if self.u_ts_constraints:
            self.regu_max = Lib.fill_extra_data(self.regu_max, years,
                                                0, frequency)
            self.regu_max = Lib.drop_extra_data(self.regu_max, years)

            self.regu_min = Lib.fill_extra_data(self.regu_min, years,
                                                0, frequency)
            self.regu_min = Lib.drop_extra_data(self.regu_min, years)

        if self.d_ts_constraints:
            self.regd_max = Lib.fill_extra_data(self.regd_max, years,
                                                0, frequency)
            self.regd_max = Lib.drop_extra_data(self.regd_max, years)

            self.regd_min = Lib.fill_extra_data(self.regd_min, years,
                                                0, frequency)
            self.regd_min = Lib.drop_extra_data(self.regd_min, years)

    def get_energy_option_up(self, mask):
        """ transform the energy option up into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.Parameter(sum(mask), value=self.eou_avg.loc[mask].values,
                             name='LF_EOU')

    def get_energy_option_down(self, mask):
        """ transform the energy option down into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.Parameter(sum(mask), value=self.eod_avg.loc[mask].values,
                             name='LF_EOD')

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                    net_ess_power, combined_rating):
        """build constraint list method for the optimization engine

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent
                generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional
                generation within the system
            net_ess_power (list, Expression): the sum of the net power of all
                the ESS in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class
                type

        Returns:
            An list of constraints for the optimization variables added to
            the system of equations
        """
        constraint_list = super().constraints(mask, load_sum, tot_variable_gen,
                                              generator_out_sum,
                                              net_ess_power, combined_rating)

        # add time series service participation constraints, if called for
        #   Reg Up Max and Reg Up Min will constrain the sum of up_ch + up_dis
        if self.u_ts_constraints:
            constraint_list += [
                cvx.NonPos(self.variables['up_ch'] + self.variables['up_dis']
                           - self.regu_max.loc[mask])
            ]
            constraint_list += [
                cvx.NonPos((-1) * self.variables['up_ch'] + (-1) * self.variables[
                    'up_dis'] + self.regu_min.loc[mask])
            ]
        #   Reg Down Max and Reg Down Min will constrain the sum down_ch+down_dis
        if self.d_ts_constraints:
            constraint_list += [
                cvx.NonPos(self.variables['down_ch'] + self.variables['down_dis']
                           - self.regd_max.loc[mask])
            ]
            constraint_list += [
                cvx.NonPos(-self.variables['down_ch'] - self.variables['down_dis']
                           + self.regd_min.loc[mask])
            ]
        return constraint_list

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals
        that are saved in the arguments of the method. Only updates the
        price signals that exist, and does not require all price signals
        needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        if self.combined_market:
            try:
                fr_price = time_series_data.loc[:, 'LF Price ($/kW)']
            except KeyError:
                pass
            else:
                self.p_regu = np.divide(fr_price, 2)
                self.p_regd = np.divide(fr_price, 2)

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
            except KeyError:
                pass
        else:
            try:
                self.p_regd = time_series_data.loc[:, 'LF Down Price ($/kW)']
            except KeyError:
                pass

            try:
                self.p_regu = time_series_data.loc[:, 'LF Up Price ($/kW)']
            except KeyError:
                pass

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
            except KeyError:
                pass

    def min_regulation_up(self):
        if self.u_ts_constraints:
            return self.regu_min
        return super().min_regulation_up()

    def min_regulation_down(self):
        if self.d_ts_constraints:
            return self.regd_min
        return super().min_regulation_down()

    def max_participation_is_defined(self):
        return self.u_ts_constraints and self.d_ts_constraints
