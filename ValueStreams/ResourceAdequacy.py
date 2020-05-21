"""
ResourceAdequacy.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .ValueStream import ValueStream
import pandas as pd
import numpy as np
import logging
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class ResourceAdequacy(ValueStream):
    """ Resource Adequacy ValueStream. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params, techs, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            dt (float): optimization timestep (hours)

        TODO: edge cases to consider -- if the event occurs at the beginning or end of an opt window  --HN
        """

        # generate the generic service object
        ValueStream.__init__(self, techs['Storage'], 'Resource Adequacy', dt)

        # add RA specific attributes
        self.days = params['days']  # number of peak events
        self.length = params['length']  # discharge duration
        self.idmode = params['idmode'].lower()  # peak selection mode
        self.dispmode = params['dispmode']  # dispatch mode
        self.capacity_rate = params['value']  # monthly RA capacity rate (length = 12)
        self.active = params['active'] == 1  # active RA timesteps (length = 8760/dt) must be boolean, not int
        self.system_load = params['system_load']  # system load profile (length = 8760/dt)
        self.dt = params['dt']  # dt for the system load profile

        # FIND THE TIME-STEPS PEAKS THAT THE RA EVENTS WILL OCCUR AROUND
        self.peak_intervals = []
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
                # 1) sort system load, during ACTIVE time-steps from largest to smallest
                max_int = year_o_system_load.loc[self.active].sort_values(ascending=False)
                # 2) keep only the first (and therefore largest) instant load per day, using an array of booleans that are True
                # for every item that has already occurred before in the index
                max_int_date = pd.Series(max_int.index.date, index=max_int.index)
                max_days = max_int.loc[~max_int_date.duplicated(keep='first')]

                # 3) select peak time-steps
                # find number of events in month where system_load is at peak during active hours:
                # select only first DAYS number of timestamps, per month
                self.peak_intervals += list(max_days.groupby(by=max_days.index.month).head(self.days).index.values)

        # DETERMINE RA EVENT INTERVALS
        event_interval = pd.Series(np.zeros(len(self.system_load)), index=self.system_load.index)
        event_start = pd.Series(np.zeros(len(self.system_load)), index=self.system_load.index)  # used to set energy constraints
        # odd intervals straddle peak & even intervals have extra interval after peak
        steps = self.length/self.dt
        if steps % 2:  # this is true if mod(steps/2) is not 0 --> if steps is odd
            presteps = np.floor_divide(steps, 2)
        else:  # steps is even
            presteps = (steps/2) - 1
        poststeps = presteps + 1

        for peak in self.peak_intervals:
            # TODO: check that this works for sub-hourly system load profiles
            first_int = peak - pd.Timedelta(presteps*self.dt, unit='h')
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

        # DETERMINE QUALIFYING COMMITMENT & ENERGY
        p_max = techs['Storage'].dis_max_rated
        energy_max = techs['Storage'].ene_max_rated
        ulsoc = techs['Storage'].ulsoc
        llsoc = techs['Storage'].llsoc

        self.qualifying_commitment = np.minimum(p_max, (energy_max*ulsoc)/self.length)
        total_time_intervals = len(self.event_intervals)

        if self.dispmode:
            # create dispatch power constraint
            # charge power should be 0, while discharge should be be the qualifying commitment for the times that correspond to the RA event

            self.charge_max_constraint = pd.Series(np.zeros(total_time_intervals), index=self.event_intervals,
                                                   name='RA Charge Max (kW)')
            self.discharge_min_constraint = pd.Series(np.repeat(self.qualifying_commitment, total_time_intervals), index=self.event_intervals,
                                                      name='RA Discharge Min (kW)')
            self.constraints = {'dis_min': Const.Constraint('dis_min', self.name, self.discharge_min_constraint),
                                'ch_max': Const.Constraint('ch_max', self.name, self.charge_max_constraint)}
        else:
            # create energy reservation constraint
            # TODO: double check to see if this needs to stack...
            # in the event of a black out -- you will not be providing resource adequacy, so no?
            qualifying_energy = self.qualifying_commitment * self.length

            # we constrain the energy to be at least the qualifying energy value at the beginning of the RA event to make sure that we
            # have enough energy to meet our promise during the entirety of the event.
            self.energy_min_constraint = pd.Series(np.repeat(qualifying_energy, len(self.event_start_times)), index=self.event_start_times,
                                                   name='RA Energy Min (kWh)')
            self.constraints = {'ene_min': Const.Constraint('ene_min', self.name, self.energy_min_constraint)}

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A tuple of a DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided), a list (of columns that remain zero), and a list (of columns that
        retain a constant value over the entire project horizon).

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        proforma, _, _ = ValueStream.proforma_report(self, opt_years, results)
        proforma[self.name + 'Capacity Payment'] = 0

        # TODO: there should be a check to make sure the commitments where actually met before including it --HN
        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.qualifying_commitment * np.sum(self.capacity_rate)

        return proforma, None, None

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
            report = pd.merge(report, self.discharge_min_constraint, how='left', on='Start Datetime')
        else:
            report = pd.merge(report, self.energy_min_constraint, how='left', on='Start Datetime')
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

