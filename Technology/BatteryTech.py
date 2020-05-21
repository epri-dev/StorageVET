"""
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .Storage import Storage
import copy
import logging
import numpy as np
import pandas as pd
import rainflow

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class BatteryTech(Storage):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, name,  params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object
        Storage.__init__(self, params['name'], params)

        self.hp = params['hp']
        self.name = params['name']

        # add degradation information
        self.cycle_life = params['cycle_life']
        self.degrade_perc = 0
        self.yearly_degrade = params['yearly_degrade'] / 100
        self.incl_cycle_degrade = bool(params['incl_cycle_degrade'])
        self.degrade_data = None

    def initialize_degredation(self, opt_agg):
        """

        Notes: Should be called once, after optimization levels are assigned, but before
        optimization loop gets called

        Args:
            opt_agg (DataFrame):

        Returns: None

        """
        self.degrade_data = pd.DataFrame(index=opt_agg.control.unique())
        # calculate current degrade_perc since installation
        if self.incl_cycle_degrade:
            start_dttm = opt_agg.index[0].to_timestamp()
            self.calc_degradation(None, self.operation_date, start_dttm)
            self.degrade_data['degrade_perc'] = self.degrade_perc
            self.degrade_data['eff_e_cap'] = self.apply_degradation()

    def calc_degradation(self, opt_period, start_dttm, end_dttm):
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            opt_period: the index of the optimization that occurred before calling this function, None if
                no optimization problem has been solved yet
            start_dttm (DateTime): Start timestamp to calculate degradation. ie. the first datetime in the optimization
                problem
            end_dttm (DateTime): End timestamp to calculate degradation. ie. the last datetime in the optimization
                problem

        A percent that represented the energy capacity degradation
        """

        # time difference between time stamps converted into years multiplied by yearly degrate rate
        if self.incl_cycle_degrade:

            # calculate degradation due to cycling iff energy values are given
            if opt_period is not None:
                energy_series = self.variables.loc[start_dttm:end_dttm, 'ene']
                # use rainflow counting algorithm to get cycle counts
                cycle_counts = rainflow.count_cycles(energy_series, ndigits=4)

                # sort cycle counts into user inputed cycle life bins
                digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],
                                                   [min(i[0]/self.ene_max_rated, 1) for i in cycle_counts], side='left')

                # sum up number of cycles for all cycle counts in each bin
                cycle_sum = self.cycle_life.loc[:, :]
                cycle_sum.loc[:, 'cycles'] = 0
                for i in range(len(cycle_counts)):
                    cycle_sum.loc[digitized_cycles[i], 'cycles'] += cycle_counts[i][1]

                # sum across bins to get total degrade percent
                # 1/cycle life value is degrade percent for each cycle
                cycle_degrade = np.dot(1/cycle_sum['Cycle Life Value'], cycle_sum.cycles)
            else:
                cycle_degrade = 0

            # add the degradation due to time passing and cycling for total degradation
            degrade_percent = cycle_degrade

            # record the degradation
            if opt_period:
                # the total degradation after optimization OPT_PERIOD must also take into account the
                # degradation that occurred before the battery was in operation (which we saved as SELF.DEGRADE_PERC)
                self.degrade_data.loc[opt_period, 'degrade_perc'] = degrade_percent + self.degrade_perc
            else:
                # if this calculation is done pre-optimization loop, then save this value as an attribute
                # self.degrade_perc = degrade_percent + self.degrade_perc
                self.degrade_perc = degrade_percent

    def apply_degradation(self, datetimes=None):
        """ Updates ene_max_rated and control constraints based on degradation percent

        Args:
            datetimes (DateTime): Vector of timestamp to recalculate control_constraints. Default is None which results in control constraints not updated

        Returns:
            Degraded energy capacity
        """

        # apply degrade percent to rated energy capacity
        new_ene_max = max(self.ulsoc*self.ene_max_rated*(1-self.degrade_perc), 0)

        # update physical constraint
        self.physical_constraints['ene_max_rated'].value = new_ene_max

        failure = None
        if datetimes is not None:
            # update control constraints
            failure = self.calculate_control_constraints(datetimes)
        if failure is not None:
            # possible that degredation caused infeasible scenario
            u_logger.error('Degradation results in infeasible scenario')
            e_logger.error('Degradation results in infeasible scenario while applying degradation for BatteryTech')
            quit()
        return new_ene_max

    def apply_past_degredation(self, mask, opt_period):
        """ Applies degradation between optimization window datasets. Calculates the new
        effective energy maximum.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            opt_period (int): optimization period for which the data applies

        """
        if self.incl_cycle_degrade:
            if opt_period:
                self.degrade_perc = self.degrade_data.iloc[max(opt_period - 1, 0)].loc['degrade_perc']

            # apply degradation to technology (affects physical_constraints['ene_max_rated'] and control constraints)
            self.degrade_data.loc[opt_period, 'eff_e_cap'] = self.apply_degradation(mask.index)

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its
            service constraints.
        """

        # create default list of constraints
        constraint_list = Storage.objective_constraints(self, variables, mask, reservations, mpc_ene)
        # add constraint that battery can not charge and discharge in the same timestep
        if self.incl_binary:
            # can only be on or off
            constraint_list += [variables['on_c'] + variables['on_d'] <= 1]

            # # when trying non binary
            # constraint_list += [0 <= on_c]
            # constraint_list += [0 <= on_d]

            # # NL formulation of binary variables
            # constraint_list += [cvx.square(on_c) - on_c == 0]
            # constraint_list += [cvx.square(on_d) - on_d == 0]

        return constraint_list

    def save_variable_results(self, variables, subs_index):
        """In addition to saving the variable results, this also makes sure that
        there is not charging and discharging at the same time

        Args:
            variables:
            subs_index:

        Returns:

        """
        super().save_variable_results(variables, subs_index)

        # check for charging and discharging in same time step
        eps = 1e-4
        if any(((abs(self.variables['ch']) >= eps) & (abs(self.variables['dis']) >= eps))):
            u_logger.warning('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')
            e_logger.warning('WARNING! non-zero charge and discharge powers found in optimization solution. Try binary formulation')

    def proforma_report(self, opt_years, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            results (DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

            Creates a dataframe with only the years that we have data for. Since we do not label the column,
            it defaults to number the columns with a RangeIndex (starting at 0) therefore, the following
            DataFrame has only one column, labeled by the int 0

        """
        pro_forma = Storage.proforma_report(self, opt_years, results)

        if self.hp > 0:
            # the value of the energy consumed by the auxiliary load (housekeeping power) is assumed to be equal to the
            # value of energy for DA ETS, real time ETS, or retail ETS.
            if results.columns.isin(['DA Price Signal ($/kWh)']).any():
                hp_cost = self.dt * -results.loc[:, 'DA Price Signal ($/kWh)'] * self.hp
                for year in opt_years:
                    year_monthly = hp_cost[hp_cost.index.year == year]
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Aux Load Cost'] = year_monthly.sum()
            if results.columns.isin(['Energy Price ($/kWh)']).any():
                hp_cost = self.dt * -results.loc[:, 'Energy Price ($/kWh)'] * self.hp
                for year in opt_years:
                    year_monthly = hp_cost[hp_cost.index.year == year]
                    pro_forma.loc[pd.Period(year=year, freq='y'), 'Aux Load Cost'] = year_monthly.sum()

        return pro_forma

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = Storage.timeseries_report(self)
        results[self.name + ' Discharge (kW)'] = self.variables['dis']
        results[self.name + ' Charge (kW)'] = self.variables['ch']
        results[self.name + ' Power (kW)'] = self.variables['dis'] - self.variables['ch']
        results[self.name + ' State of Energy (kWh)'] = self.variables['ene']

        try:
            energy_rate = self.ene_max_rated.value
        except AttributeError:
            energy_rate = self.ene_max_rated

        results['Battery SOC (%)'] = self.variables['ene'] / energy_rate

        return results
