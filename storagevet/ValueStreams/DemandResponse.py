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
DemandResponse.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import pandas as pd
import cvxpy as cvx
import numpy as np
from storagevet.ErrorHandling import *
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib

SATURDAY = 5


class DemandResponse(ValueStream):
    """ Demand response program participation. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters

        """

        # generate the generic service object
        ValueStream.__init__(self, 'Demand Response', params)

        # add dr specific attributes to object
        self.days = params['days']
        self.length = params['length']  # length of an event
        self.weekend = params['weekend']
        self.start_hour = params['program_start_hour']
        self.end_hour = params['program_end_hour']  # last hour of the program
        self.day_ahead = params['day_ahead']  # indicates whether event is scheduled in real-time or day ahead

        # timeseries data
        self.system_load = params['system_load']
        self.months = params['dr_months'] == 1
        self.cap_commitment = params['dr_cap']  # this is the max capacity a user is willing to provide

        # monthly data
        self.cap_monthly = params['cap_monthly']
        self.cap_price = params['cap_price']
        self.ene_price = params['ene_price']

        # the following attributes will be used to save values during analysis
        self.qc = None
        self.qe = None
        self.charge_max_constraint = pd.Series()
        self.discharge_min_constraint = pd.Series()
        self.energy_min_constraint = pd.Series()
        self.possible_event_times = None

    def grow_drop_data(self, years, frequency, load_growth):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation


        """
        # timeseries data
        self.system_load = Lib.fill_extra_data(self.system_load, years, load_growth, frequency)
        self.system_load = Lib.drop_extra_data(self.system_load, years)

        self.months = Lib.fill_extra_data(self.months, years, 0, frequency)
        self.months = Lib.drop_extra_data(self.months, years)

        self.cap_commitment = Lib.fill_extra_data(self.cap_commitment, years, 0, frequency)
        self.cap_commitment = Lib.drop_extra_data(self.cap_commitment, years)

        # monthly data
        self.cap_monthly = Lib.fill_extra_data(self.cap_monthly, years, 0, 'M')
        self.cap_monthly = Lib.drop_extra_data(self.cap_monthly, years)

        self.cap_price = Lib.fill_extra_data(self.cap_price, years, 0, 'M')
        self.cap_price = Lib.drop_extra_data(self.cap_price, years)

        self.ene_price = Lib.fill_extra_data(self.ene_price, years, 0, 'M')
        self.ene_price = Lib.drop_extra_data(self.ene_price, years)

    def calculate_system_requirements(self, der_lst):
        """ Calculate the system requirements that must be meet regardless of what other value streams are active
        However these requirements do depend on the technology that are active in our analysis

        Args:
            der_lst (list): list of the initialized DERs in our scenario

        """
        max_discharge_possible = self.qualifying_commitment(der_lst, self.length)

        if self.day_ahead:
            # if events are scheduled the "day ahead", exact time of the event is known and we can plan accordingly
            indx_dr_days = self.day_ahead_event_scheduling()
        else:
            # power reservations instead of absolute constraints
            # if events are scheduled the "Day of", or if the start time of the event is uncertain, apply at every possible start
            indx_dr_days = self.day_of_event_scheduling()

        qc = np.minimum(self.cap_commitment.loc[indx_dr_days].values, max_discharge_possible)
        self.qc = pd.Series(qc, index=indx_dr_days, name='DR Discharge Min (kW)')
        self.possible_event_times = indx_dr_days

        if self.day_ahead:
            self.charge_max_constraint = pd.Series(np.zeros(len(indx_dr_days)), index=indx_dr_days, name='DR Charge Max (kW)')
            self.discharge_min_constraint = pd.Series(qc, index=indx_dr_days, name='DR Discharge Min (kW)')
            self.system_requirements += [Requirement('discharge', 'min', self.name, self.discharge_min_constraint),
                                         Requirement('charge', 'max', self.name, self.charge_max_constraint)]
        else:
            self.qualifying_energy()

    def day_ahead_event_scheduling(self):
        """ If Dr events are scheduled with the STORAGE operator the day before the event, then the operator knows
        exactly when these events will occur.

        We need to make sure that the storage can perform, IF called upon, by making the battery can discharge atleast enough
        to meet the qualifying capacity and reserving enough energy to meet the full duration of the event

        START_HOUR is required. A user must also provide either END_HOUR or LENGTH

        Returns: index for when the qualifying capacity must apply

        """
        if not self.end_hour and not self.length:
            TellUser.error('Demand Response: You must provide either program_end_hour or length for day ahead scheduling')
            raise ParameterError('Demand Response: You must provide either program_end_hour or length for day ahead scheduling')
        if self.end_hour and self.length:
            # require that LENGTH < END_HOUR - START_HOUR
            if self.length != self.end_hour - self.start_hour + 1:
                TellUser.error('Demand Response: event length is not program_end_hour - program_start_hour.'
                               + 'Please provide either program_end_hour or length for day ahead scheduling')
                raise ParameterError('Demand Response: event length is not program_end_hour - program_start_hour.'
                                + 'Please provide either program_end_hour or length for day ahead scheduling')

        if not self.end_hour:
            self.end_hour = self.start_hour + self.length - 1

        elif not self.length:
            self.length = self.end_hour - self.start_hour + 1

        index = self.system_load.index
        he = index.hour + 1  # hour ending

        ##########################
        # FIND DR EVENTS: system load -> group by date -> filter by DR hours -> sum system load energy -> filter by top n days
        ##########################

        # dr program is active based on month and if hour is in program hours
        active = self.months & (he >= self.start_hour) & (he <= self.end_hour)

        # remove weekends from active datetimes if dr_weekends is False
        if not self.weekend:
            active = active & (active.index.weekday < SATURDAY).astype('int64')

        # 1) system load, during ACTIVE time-steps from largest to smallest
        load_during_active_events = self.system_load.loc[active]

        # 2) system load is groupby by date and summed and multiplied by DT
        sum_system_load_energy = load_during_active_events.groupby(by=load_during_active_events.index.date).sum() * self.dt

        # 3) sort the energy per event and select peak time-steps
        # find number of events in month where system_load is at peak during active hours: select only first DAYS number of timestamps, per month
        disp_days = sum_system_load_energy.sort_values(ascending=False)[:self.days]

        # create a mask that is true when ACTIVE is true and the date is in DISP_DAYS.INDEX
        active_event_mask = pd.Series(np.repeat(False, len(index)), index=index)
        for date in disp_days.index:
            active_event_mask = (index.date == date) & active | active_event_mask
        # create index for power constraint
        indx_dr_days = active_event_mask.loc[active_event_mask].index

        return indx_dr_days

    def day_of_event_scheduling(self):
        """ If the DR events are scheduled the day of, then the STORAGE operator must prepare for an event to occur
        everyday, to start at any time between the program START_HOUR and ending on END_HOUR for the duration of
        LENGTH.

        In this case there must be a power reservations because the storage might or might not be called, so there must
        be enough power capacity to provide for all services.

        Returns: index for when the qualifying capacity must apply

        """
        if self.end_hour != 'nan' and self.length != 'nan':
            # require that LENGTH < END_HOUR - START_HOUR
            if self.length > self.end_hour - self.start_hour + 1:
                TellUser.error('Demand Response: event length is not program_end_hour - program_start_hour.'
                               + 'Please provide either program_end_hour or length for day ahead scheduling')
                raise ParameterError('Demand Response: event length is not program_end_hour - program_start_hour.'
                                     + 'Please provide either program_end_hour or length for day ahead scheduling')

        ##########################
        # FIND DR EVENTS
        ##########################
        index = self.system_load.index
        he = index.hour + 1

        # dr program is active based on month and if hour is in program hours
        active = self.months & (he >= self.start_hour) & (he <= self.end_hour)

        # remove weekends from active datetimes if dr_weekends is False
        if not self.weekend:
            active = active & (active.index.weekday < SATURDAY).astype('int64')

        return active.loc[active].index

    @staticmethod
    def qualifying_commitment(der_lst, length):
        """

        Args:
            der_lst (list): list of the initialized DERs in our scenario
            length (int): length of the event

        NOTE: RA has this same method too  -HN

        """
        qc = sum(der_instance.qualifying_capacity(length) for der_instance in der_lst)
        return qc

    def qualifying_energy(self):
        """ Calculated the qualifying energy to be able to participate in an event.

        This function should be called after calculating the qualifying commitment.

        """
        qe = self.qc * self.length  # qualifying energy timeseries dataframe

        # only apply energy constraint whenever an event could start. So we need to get rid of the last (SELF.LENGTH*SELF.DT) - 1 timesteps per event
        last_start = self.end_hour - self.length  # this is the last time that an event can start
        mask = qe.index.hour <= last_start

        self.qe = qe.loc[mask]

    def p_reservation_discharge_up(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        if not self.day_ahead:
            # make sure we will be able to discharge if called upon (in addition to other market services)
            subs = mask.loc[mask]
            dis_reservation = pd.Series(np.zeros(sum(mask)), index=subs.index)
            subs_qc = self.qc.loc[self.qc.index.isin(subs.index)]
            if not subs_qc.empty:
                dis_reservation.update(subs_qc)
            down = cvx.Parameter(shape=sum(mask), value=dis_reservation.values, name='DischargeResDR')
        else:
            down = super().p_reservation_discharge_up(mask)
        return down

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
        proforma[self.name + ' Energy Payment'] = 0

        energy_displaced = results.loc[self.possible_event_times, 'Total Storage Power (kW)']
        energy_displaced += results.loc[self.possible_event_times, 'Total Generation (kW)']

        for year in opt_years:
            year_cap_price = self.cap_price.loc[self.cap_price.index.year == year]
            year_monthly_cap = self.cap_monthly.loc[self.cap_monthly.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Capacity Payment'] = \
                np.sum(np.multiply(year_monthly_cap, year_cap_price))

            if self.day_ahead:
                # in our "day of" event notification: the battery does not actually dispatch to
                # meet a DR event, so no $ is awarded
                year_subset = energy_displaced[energy_displaced.index.year == year]
                year_ene_price = self.ene_price.loc[self.ene_price.index.year == year]
                energy_payment = 0
                for month in range(1, 13):
                    energy_payment = \
                        np.sum(year_subset.loc[month == year_subset.index.month]) * year_ene_price.loc[year_ene_price.index.month == month].values
                proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Energy Payment'] = \
                    energy_payment * self.dt
        # apply inflation rates
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))

        return proforma

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.system_load.index)
        report.loc[:, "System Load (kW)"] = self.system_load
        if self.day_ahead:
            report.loc[:, self.discharge_min_constraint.name] = 0
            report.update(self.discharge_min_constraint)
        else:
            report.loc[:, 'DR Possible Event (y/n)'] = False
            report.loc[self.possible_event_times, 'DR Possible Event (y/n)'] = True
        return report

    def monthly_report(self):
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """

        monthly_financial_result = pd.DataFrame({'DR Capacity Price ($/kW)': self.cap_price.values}, index=self.cap_price.index)
        monthly_financial_result.loc[:, 'DR Energy Price ($/kWh)'] = self.ene_price
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
            self.cap_price = monthly_data.loc[:, 'DR Capacity Price ($/kW)']
        except KeyError:
            pass

        try:
            self.ene_price = monthly_data.loc[:, 'DR Energy Price ($/kWh)']
        except KeyError:
            pass
