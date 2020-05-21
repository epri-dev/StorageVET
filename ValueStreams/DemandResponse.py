"""
DemandResponse.py

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
import cvxpy as cvx
import numpy as np
import logging
try:
    import Constraint as Const
except ModuleNotFoundError:
    import storagevet.Constraint as Const
# import heapq

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class DemandResponse(ValueStream):
    """ Demand response program participation. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, techs, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            dt (float): optimization timestep (hours)

        """

        # generate the generic service object
        ValueStream.__init__(self, techs['Storage'], 'Demand Response', dt)

        # add dr specific attributes to object
        self.dt = params['dt']
        self.days = params['days']
        self.length = params['length']  # length of an event
        self.weekend = params['weekend']
        self.start_hour = params['program_start_hour']
        self.end_hour = params['program_end_hour']  # or this is just start + length
        self.day_ahead = params['day_ahead']  # indicates whether event is scheduled in real-time or day ahead

        # timeseries data
        self.system_load = params['system_load']
        self.months = params['dr_months'] == 1
        self.cap_commitment = params['dr_cap']  # this is the max capacity a user is willing to provide

        # monthly data
        self.cap_monthly = params['cap_monthly']
        self.cap_price = params['cap_price']
        self.ene_price = params['ene_price']

        # these need to be set based on if day-ahead or
        self.qc = None
        self.qe = None
        self.charge_max_constraint = 0
        self.discharge_min_constraint = 0
        self.energy_min_constraint = 0

        if self.day_ahead:
            # if events are scheduled the "day ahead", exact time of the event is known and we can plan accordingly
            indx_dr_days = self.day_ahead_event_scheduling()
            self.qualifying_commitment(indx_dr_days)
            self.charge_max_constraint = pd.Series(np.zeros(len(indx_dr_days)), index=indx_dr_days, name='DR Charge Max (kW)')
            self.discharge_min_constraint = pd.Series(self.qc, index=indx_dr_days, name='DR Discharge Min (kW)')
            # self.energy_min_constraint = pd.Series(self.qe, index=indx_dr_days, name='DR Energy Min (kW)')
            self.constraints = {'dis_min': Const.Constraint('dis_min', self.name, self.discharge_min_constraint),
                                'ch_max': Const.Constraint('ch_max', self.name, self.charge_max_constraint)}

            self.possible_event_times = indx_dr_days

        else:
            # power reservations instead of absolute constraints
            # if events are scheduled the "Day of", or if the start time of the event is uncertain, apply at every possible start
            self.possible_event_times = self.day_of_event_scheduling()
            self.qualifying_commitment(self.possible_event_times)
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
            e_logger.error('Demand Response: You must provide either program_end_hour or length for day ahead scheduling')
            raise Exception('Demand Response: You must provide either program_end_hour or length for day ahead scheduling')
        if self.end_hour and self.length:
            # require that LENGTH < END_HOUR - START_HOUR
            if self.length != self.end_hour - self.start_hour:
                e_logger.error('Demand Response: event length is not program_end_hour - program_start_hour.'
                               + 'Please provide either program_end_hour or length for day ahead scheduling')
                raise Exception('Demand Response: event length is not program_end_hour - program_start_hour.'
                                + 'Please provide either program_end_hour or length for day ahead scheduling')

        if not self.end_hour:
            self.end_hour = self.start_hour + self.length

        elif not self.length:
            self.length = self.end_hour - self.start_hour

        index = self.system_load.index
        he = index.hour + 1

        ##########################
        # FIND DR EVENTS: system load -> group by date -> filter by DR hours -> sum system load energy -> filter by top n days
        ##########################

        # dr program is active based on month and if hour is in program hours
        active = self.months & (he > self.start_hour) & (he <= self.end_hour)

        # remove weekends from active datetimes if dr_weekends is False
        SATURDAY = 5
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
            if self.length > self.end_hour - self.start_hour:
                e_logger.error('Demand Response: event length is not program_end_hour - program_start_hour.'
                               + 'Please provide either program_end_hour or length for day ahead scheduling')
                raise Exception('Demand Response: event length is not program_end_hour - program_start_hour.'
                                + 'Please provide either program_end_hour or length for day ahead scheduling')

        ##########################
        # FIND DR EVENTS
        ##########################
        index = self.system_load.index
        he = index.hour + 1

        # dr program is active based on month and if hour is in program hours
        active = self.months & (he > self.start_hour) & (he <= self.end_hour)

        # remove weekends from active datetimes if dr_weekends is False
        SATURDAY = 5
        if not self.weekend:
            active = active & (active.index.weekday < SATURDAY).astype('int64')

        return active.loc[active].index

    def qualifying_commitment(self, dr_event_times):
        """

        Args:
            dr_event_times (Index): Index of dates that contain possible or certain DR events

        """
        # collect battery attributes
        p_max = self.storage.dis_max_rated
        energy = self.storage.ene_max_rated  # P_MAX * duration of battery
        min_power_bat = np.minimum(p_max, energy / self.length)

        qc = np.minimum(self.cap_commitment.loc[dr_event_times].values, min_power_bat)  # qualifying commitment timeseries subset dataframe
        self.qc = pd.Series(qc, index=dr_event_times, name='DR Discharge Min (kW)')

    def qualifying_energy(self):
        """ Calculated the qualifying energy to be able to participate in an event.

        This function should be called after calculating the qualifying commitment.

        """
        qe = self.qc * self.length  # qualifying energy timeseries dataframe

        # only apply energy constraint whenever an event could start. So we need to get rid of the last (SELF.LENGTH*SELF.DT) - 1 timesteps per event
        last_start = self.end_hour - self.length  # this is the last time that an event can start
        mask = qe.index.hour <= last_start

        self.qe = qe.loc[mask]

    def power_ene_reservations(self, opt_vars, mask):
        """ Determines power and energy reservations required at the end of each timestep for the service to be provided.
        Additionally keeps track of the reservations per optimization window so the values maybe accessed later.

        Args:
            opt_vars (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A power reservation and a energy reservation array for the optimization window--
            C_max, C_min, D_max, D_min, E_upper, E, and E_lower (in that order)
        """
        if self.day_ahead:
            return ValueStream.power_ene_reservations(self, opt_vars, mask)
        else:
            eta = self.storage.rte
            size = opt_vars['ene'].shape
            subs = mask.loc[mask]
            dis_min_reservation = pd.Series(np.zeros(size), index=subs.index)

            subs_qc = self.qc.loc[self.qc.index.isin(subs.index)]

            if not subs_qc.empty:
                dis_min_reservation.update(subs_qc)

            # calculate reservations
            c_max = 0
            c_min = 0
            d_min = cvx.Parameter(shape=size, value=dis_min_reservation.values, name='Dis min reservation DR')
            d_max = 0
            e_upper = cvx.Parameter(shape=size, value=np.zeros(size), name='e_upper_dr')
            e = cvx.Parameter(shape=size, value=np.zeros(size), name='e_dr')
            e_lower = cvx.Parameter(shape=size, value=np.zeros(size), name='e_lower_dr')

            # save reservation for optimization window
            self.e.append(e)
            self.e_lower.append(e_lower)
            self.e_upper.append(e_upper)
            self.c_max.append(c_max)
            self.c_min.append(c_min)
            self.d_max.append(d_max)
            self.d_min.append(d_min)
            return [c_max, c_min, d_max, d_min], [e_upper, e, e_lower]

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
        proforma, zero_col, _ = ValueStream.proforma_report(self, opt_years, results)
        proforma[self.name + ' Capacity Payment'] = 0
        proforma[self.name + ' Energy Payment'] = 0

        energy_displaced = results.loc[self.possible_event_times, 'Total Storage Power (kW)']
        try:
            # This will need to be the sum of power out of all gensets
            energy_displaced += results.loc[self.possible_event_times, 'Diesel Generation (kW)']
        except KeyError:
            pass

        # TODO: there should be a check to make sure the commitments where actually met before including it --HN
        for year in opt_years:
            year_cap_price = self.cap_price.loc[self.cap_price.index.year == year]
            year_monthly_cap = self.cap_monthly.loc[self.cap_monthly.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Capacity Payment'] = np.sum(np.multiply(year_monthly_cap, year_cap_price))

            # in our "day of" event notification: the battery does not actually dispatch to meet a DR event, so no $ is awarded
            if self.day_ahead:
                year_subset = energy_displaced[energy_displaced.index.year == year]
                year_ene_price = self.ene_price.loc[self.ene_price.index.year == year]
                energy_payment = 0
                for month in range(1, 13):
                    energy_payment = np.sum(year_subset.loc[month == year_subset.index.month]) * year_ene_price.loc[year_ene_price.index.month == month].values
                proforma.loc[pd.Period(year=year, freq='y'), self.name + ' Energy Payment'] = energy_payment * self.dt
            else:
                zero_col = [self.name + ' Energy Payment']

        return proforma, zero_col, None

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
