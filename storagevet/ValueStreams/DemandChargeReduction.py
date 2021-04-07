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
DemandChargeReduction.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import numpy as np
import cvxpy as cvx
import pandas as pd
import sys
from storagevet.Finances import Financial
from storagevet.ErrorHandling import *
import copy
import time

SATURDAY = 5


class DemandChargeReduction(ValueStream):
    """ Retail demand charge reduction. A behind the meter service.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
        """
        ValueStream.__init__(self, 'DCM', params)
        # self.demand_rate = params['rate']
        self.tariff = params['tariff']
        self.billing_period = params['billing_period']
        self.growth = params['growth']/100

        self.billing_period_bill = pd.DataFrame()
        self.monthly_bill = pd.DataFrame()

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        This function adds billing periods to the tariff that match the given year's structure, but the values have
        a growth rate applied to them. Then it lists them within self.billing_period.

        """
        data_year = self.billing_period.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                years = yr.year - source_year.year

                first_day = '1/1/' + str(yr.year)
                last_day = '1/1/' + str(yr.year + 1)

                new_index = pd.date_range(start=first_day, end=last_day, freq=frequency, closed='left')
                size = new_index.size

                # make new tariff with charges that have increase with user-defined growth rate
                add_tariff = self.tariff.reset_index()
                add_tariff.loc[:, 'Value'] = self.tariff['Value'].values*(1+self.growth)**years
                add_tariff.loc[:, 'Billing Period'] = self.tariff.index + self.tariff.index.max()
                add_tariff = add_tariff.set_index('Billing Period', drop=True)
                # Build Energy Price Vector based on the new year
                temp = pd.DataFrame(index=new_index)
                weekday = new_index.weekday
                he = (new_index + pd.Timedelta('1s')).hour + 1

                billing_period = [[] for _ in range(size)]

                for p in add_tariff.index:
                    # edit the pricedf energy price and period values for all of the periods defined
                    # in the tariff input file
                    bill = add_tariff.loc[p, :]
                    mask = Financial.create_bill_period_mask(bill, temp.index.month, he, weekday)
                    if bill['Charge'].lower() == 'demand':
                        for i, true_false in enumerate(mask):
                            if true_false:
                                billing_period[i].append(p)
                billing_period = pd.Series(billing_period, dtype='object', index=temp.index)

                # ADD CHECK TO MAKE SURE ENERGY PRICES ARE THE SAME FOR EACH OVERLAPPING BILLING PERIOD
                # Check to see that each timestep has a period assigned to it
                if not billing_period.apply(len).all():
                    TellUser.error('The billing periods in the input file do not partition the year. '
                                   + 'Please check the tariff input file')
                    raise TariffError('The billing periods in the input file do not partition the year')
                self.tariff = pd.concat([self.tariff, add_tariff], sort=True)
                self.billing_period = pd.concat([self.billing_period, billing_period], sort=True)

    def objective_function(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the system. [= charge - discharge]
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            A dictionary with the portion of the objective function that it affects, labeled by the expression's key. Default is to return {}.
        """
        start = time.time()
        total_demand_charges = 0
        net_load = load_sum + net_ess_power + (-1)*generator_out_sum + (-1)*tot_variable_gen
        sub_billing_period = self.billing_period.loc[mask]
        # determine and add demand charges monthly
        months = sub_billing_period.index.to_period('M')
        for mo in months.unique():
            # array of booleans; true for the selected month
            monthly_mask = (sub_billing_period.index.month == mo.month)

            # select the month's billing period data
            month_sub_billing_period = sub_billing_period.loc[monthly_mask]

            # set of unique billing periods in the selected month
            pset = {item for sublist in month_sub_billing_period for item in sublist}

            # determine the index what has the first True value in the array of booleans
            # (the index of the timestep that corresponds to day 1 and hour 0 of the month)
            first_true = np.nonzero(monthly_mask)[0][0]

            for per in pset:
                # Add demand charge calculation for each applicable billing period (PER) within the selected month

                # get an array that is True only for the selected month
                billing_per_mask = copy.deepcopy(monthly_mask)

                for i in range(first_true, first_true + len(month_sub_billing_period)):
                    # loop through only the values that are 'True' (which should all be in order because
                    # the array should be sorted by datetime index) as they represent a month

                    # reassign the value at I to be whether PER applies to the time corresponding to I
                    # (reassign each 'True' value to 'False' if the billing period does not apply to that timestep)
                    billing_per_mask[i] = per in sub_billing_period.iloc[i]

                # add a demand charge for each billing period in a month (for every month being optimized)
                if np.all(billing_per_mask):
                    total_demand_charges += self.tariff.loc[per, 'Value'] * annuity_scalar * cvx.max(net_load)
                else:
                    total_demand_charges += self.tariff.loc[per, 'Value'] * annuity_scalar * cvx.max(net_load[billing_per_mask])
        TellUser.debug(f'time took to make demand charge term: {time.time() - start}')
        return {self.name: total_demand_charges}

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.billing_period.index)
        report.loc[:, 'Demand Charge Billing Periods'] = self.billing_period
        return report

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, **kwargs):
        """ Calculates any service related dataframe that is reported to the user.

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        return {'demand_charges': self.tariff}
