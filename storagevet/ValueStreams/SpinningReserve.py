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
SpinningReserve.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.MarketServiceUp import MarketServiceUp
import cvxpy as cvx
import storagevet.Library as Lib


class SpinningReserve(MarketServiceUp):
    """ Spinning Reserve. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
        """
        super(SpinningReserve, self).__init__('SR', 'Spinning Reserve', params)
        self.ts_constraints = params.get('ts_constraints', False)
        if self.ts_constraints:
            self.max = params['max']
            self.min = params['min']

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation


        """
        super().grow_drop_data(years, frequency, load_growth)
        if self.ts_constraints:
            self.max = Lib.fill_extra_data(self.max, years, self.growth, frequency)
            self.max = Lib.drop_extra_data(self.max, years)

            self.min = Lib.fill_extra_data(self.min, years, self.growth, frequency)
            self.min = Lib.drop_extra_data(self.min, years)

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class type

        Returns:
            An empty list (for aggregation of later constraints)
        """
        constraint_list = super().constraints(mask, load_sum, tot_variable_gen,
                                              generator_out_sum, net_ess_power, combined_rating)
        # add time series service participation constraint, if called for
        #   Max and Min will constrain the sum of ch_less and dis_more
        if self.ts_constraints:
            constraint_list += \
                [cvx.NonPos(self.variables['ch_less'] +
                            self.variables['dis_more'] - self.max.loc[mask])]
            constraint_list += \
                [cvx.NonPos(-self.variables['ch_less'] -
                            self.variables['dis_more'] + self.min.loc[mask])]

        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = super().timeseries_report()
        if self.ts_constraints:
            report.loc[:, self.max.name] = self.max
            report.loc[:, self.min.name] = self.min
        return report

    def min_regulation_down(self):
        if self.ts_constraints:
            return self.min
        return super().min_regulation_down()

    def max_participation_is_defined(self):
        return hasattr(self, 'max')
