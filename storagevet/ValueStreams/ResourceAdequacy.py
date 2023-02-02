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
ResourceAdequacy.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import pandas as pd
import numpy as np
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib


class ResourceAdequacy(ValueStream):
    """ Resource Adequacy ValueStream. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters

        """

        # generate the generic service object
        ValueStream.__init__(self, 'Resource Adequacy', params)

        # add RA specific attributes
        self.days = params['days']  # number of peak events
        self.length = params['length']  # discharge duration
        self.idmode = params['idmode'].lower()  # peak selection mode
        self.dispmode = params['dispmode']  # dispatch mode
        self.capacity_rate = params['value']  # monthly RA capacity rate (length = 12)
        if 'active hours' in self.idmode:
            self.active = params['active'] == 1  # active RA timesteps (length = 8760/dt) must be boolean, not int
        self.system_load = params['system_load']  # system load profile (length = 8760/dt)
        self.growth = params['growth'] / 100  # growth rate of RA prices, convert from % to decimal

        # initialize the following atrributes to be set later
        self.peak_intervals = []
        self.event_intervals = None
        self.event_start_times = None
        self.der_dispatch_discharge_min_constraint = None
        self.energy_min_constraint = None
        self.qc = 0

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        # timeseries data
        self.system_load = Lib.fill_extra_data(self.system_load, years, load_growth, frequency)
        self.system_load = Lib.drop_extra_data(self.system_load, years)

        if 'active hours' in self.idmode:
            self.active = Lib.fill_extra_data(self.active, years, 0, frequency)
            self.active = Lib.drop_extra_data(self.active, years)
            self.active = self.active == 1

        # monthly data
        self.capacity_rate = Lib.fill_extra_data(self.capacity_rate, years, 0, 'M')
        self.capacity_rate = Lib.drop_extra_data(self.capacity_rate, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        self.find_system_load_peaks()
        self.schedule_events()

        self.qc = self.qualifying_commitment(der_lst, self.length)
        total_time_intervals = len(self.event_intervals)

        if self.dispmode:
            # create dispatch power constraint
            # net power should be be the qualifying commitment for the times that correspond to the RA event

            self.der_dispatch_discharge_min_constraint = pd.Series(np.repeat(self.qc, total_time_intervals), index=self.event_intervals,
                                                      name='RA Discharge Min (kW)')
            self.system_requirements += [Requirement('der dispatch discharge', 'min', self.name, self.der_dispatch_discharge_min_constraint)]
        else:
            # create energy reservation constraint
            qualifying_energy = self.qc * self.length

            # we constrain the energy to be at least the qualifying energy value at the beginning of the RA event to make sure that we
            # have enough energy to meet our promise during the entirety of the event.
            self.energy_min_constraint = pd.Series(np.repeat(qualifying_energy, len(self.event_start_times)), index=self.event_start_times,
                                                   name='RA Energy Min (kWh)')
            self.system_requirements.append(Requirement('energy', 'min', self.name, self.energy_min_constraint))

    def find_system_load_peaks(self):
        """ Find the time-steps that load peaks occur on. The RA events will occur around these.

        This method edits the PEAK_INTERVALS attribute

        """
        for year in self.system_load.index.year.unique():
            year_o_system_load = self.system_load.loc[self.system_load.index.year == year]
            if self.idmode == 'peak by year':
                # 1) sort system load from largest to smallest
                max_int = year_o_system_load.sort_values(ascending=False)
                # 2) keep only the first (and therefore largest) instant load per day, using an array of booleans that are True
                # for every item that has already occurred before in the index
                max_int_date = pd.Series(max_int.index.date, index=max_int.index)
                max_days = max_int.loc[~max_int_date.duplicated(keep='first')]

                # 3) select peak time-steps
                # find ra_events number of events in year where system_load is at peak:
                # select only the first DAYS number of timestamps
                self.peak_intervals += list(max_days.index[:self.days].values)

            elif self.idmode == 'peak by month':
                # 1) sort system load from largest to smallest
                max_int = year_o_system_load.sort_values(ascending=False)
                # 2) keep only the first (and therefore largest) instant load per day, using an array of booleans that are True
                # for every item that has already occurred before in the index
                max_int_date = pd.Series(max_int.index.date, index=max_int.index)
                max_days = max_int.loc[~max_int_date.duplicated(keep='first')]

                # 3) select peak time-steps
                # find number of events in month where system_load is at peak:
                # select only the first DAYS number of timestamps, per month
                self.peak_intervals += list(max_days.groupby(by=max_days.index.month).head(self.days).index.values)

            elif self.idmode == 'peak by month with active hours':
                active_year_sub = self.active[self.system_load.index.year == year]
                # 1) sort system load, during ACTIVE time-steps from largest to smallest
                max_int = year_o_system_load.loc[active_year_sub].sort_values(ascending=False)
                # 2) keep only the first (and therefore largest) instant load per day, using an array of booleans that are True
                # for every item that has already occurred before in the index
                max_int_date = pd.Series(max_int.index.date, index=max_int.index)
                max_days = max_int.loc[~max_int_date.duplicated(keep='first')]

                # 3) select peak time-steps
                # find number of events in month where system_load is at peak during active hours:
                # select only first DAYS number of timestamps, per month
                self.peak_intervals += list(max_days.groupby(by=max_days.index.month).head(self.days).index.values)

    def schedule_events(self):
        """ Determines RA event intervals (the times for which the event will be occurring) and event start times.

        TODO: edge cases to consider -- if the event occurs at the beginning or end of an opt window  --HN
        TODO: check that this works for sub-hourly system load profiles

        """
        # DETERMINE RA EVENT INTERVALS
        event_interval = pd.Series(np.zeros(len(self.system_load)), index=self.system_load.index)
        event_start = pd.Series(np.zeros(len(self.system_load)), index=self.system_load.index)  # used to set energy constraints
        # odd intervals straddle peak & even intervals have extra interval after peak
        steps = self.length / self.dt
        if steps % 2:  # this is true if mod(steps/2) is not 0 --> if steps is odd
            presteps = np.floor_divide(steps, 2)
        else:  # steps is even
            presteps = (steps / 2) - 1
        poststeps = presteps + 1

        for peak in self.peak_intervals:
            first_int = peak - pd.Timedelta(presteps * self.dt, unit='h')
            last_int = peak + pd.Timedelta(poststeps * self.dt, unit='h')

            # handle edge RA event intervals
            if first_int < event_interval.index[0]:  # RA event starts before the first time-step in the system load
                first_int = event_interval.index[0]
            if last_int > event_interval.index[-1]:  # RA event ends after the last time-step in the system load
                last_int = event_interval.index[-1]

            event_range = pd.date_range(start=first_int, end=last_int, periods=steps)
            event_interval.loc[event_range] = 1
            event_start.loc[first_int] = 1
        self.event_intervals = self.system_load[event_interval == 1].index
        self.event_start_times = self.system_load[event_start == 1].index

    @staticmethod
    def qualifying_commitment(der_lst, length):
        """

        Args:
            der_lst (list): list of the initialized DERs in our scenario
            length (int): length of the event

        NOTE: DR has this same method too  -HN

        """
        qc = sum(der_instance.qualifying_capacity(length) for der_instance in der_lst)
        return qc

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
        proforma[self.name + ' Capacity Payment'] = 0

        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.qc * np.sum(self.capacity_rate)
        # apply inflation rates
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))
        proforma = fill_forward_func(proforma, self.growth)
        return proforma

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.system_load.index)
        report.loc[:, "System Load (kW)"] = self.system_load
        report.loc[:, 'RA Event (y/n)'] = False
        report.loc[self.event_intervals, 'RA Event (y/n)'] = True
        if self.dispmode:
            report = pd.merge(report, self.der_dispatch_discharge_min_constraint, how='left', on='Start Datetime (hb)')
        else:
            report = pd.merge(report, self.energy_min_constraint, how='left', on='Start Datetime (hb)')
        return report

    def monthly_report(self):
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """

        monthly_financial_result = pd.DataFrame({'RA Capacity Price ($/kW)': self.capacity_rate}, index=self.capacity_rate.index)
        monthly_financial_result.index.names = ['Year-Month']

        return monthly_financial_result

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.capacity_rate = monthly_data.loc[:, 'RA Capacity Price ($/kW)']
        except KeyError:
            pass
