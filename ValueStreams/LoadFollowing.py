"""
LoadFollowing.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Thien Nguyen']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .ValueStream import ValueStream
import cvxpy as cvx
import pandas as pd
import logging
import numpy as np
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class LoadFollowing(ValueStream):
    """ Load Following. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): time series timestep (hours)
        """
        ValueStream.__init__(self, tech, 'LF', dt)
        self.price = params['energy_price']
        self.price_lf_up = params['lf_up_price']
        self.price_lf_do = params['lf_do_price']
        self.offset = params['lf_offset']
        self.price_growth = params['growth']
        self.dt = dt
        self.ku = params['ku']
        self.ku_max = 0
        self.ku_min = 0
        self.kd = params['kd']
        self.kd_max = 0
        self.kd_min = 0
        self.combined_market = params['CombinedMarket']

        if self.dt > 0.25:
            e_logger.warning("WARNING: using Load Following Service and time series timestep is greater than 15 min.")
            u_logger.warning("WARNING: using Load Following Service and time series timestep is greater than 15 min.")

        self.variable_names = {'lf_up_c', 'lf_do_c', 'lf_up_d', 'lf_do_d'}
        self.variables = pd.DataFrame(columns=self.variable_names)

    def objective_function(self, variables, mask, load, generation, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables. Saves generated expression to
        within the class.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            The portion of the objective function that it affects. This can be passed into the cvxpy solver.
            Returns costs - benefits
        """
        size = sum(mask)
        p_lf_up = cvx.Parameter(size, value=self.price_lf_up.loc[mask].values, name='price_lf_up')
        p_lf_do = cvx.Parameter(size, value=self.price_lf_do.loc[mask].values, name='price_lf_do')
        p_ene = cvx.Parameter(size, value=self.price.loc[mask].values, name='price')

        lf_up_c_cap = cvx.sum(-variables['lf_up_c'] * p_lf_up)
        lf_up_c_ene = cvx.sum(-variables['lf_up_c'] * self.ku * p_ene * self.dt)

        lf_up_d_cap = cvx.sum(-variables['lf_up_d'] * p_lf_up)
        lf_up_d_ene = cvx.sum(-variables['lf_up_d'] * self.ku * p_ene * self.dt)

        lf_do_c_cap = cvx.sum(-variables['lf_do_c'] * p_lf_do)
        lf_do_c_ene = cvx.sum(variables['lf_do_c'] * self.kd * p_ene * self.dt)

        lf_do_d_cap = cvx.sum(-variables['lf_do_d'] * p_lf_do)
        lf_do_d_ene = cvx.sum(variables['lf_do_d'] * self.kd * p_ene * self.dt)

        return {'lf_up_c_cap': lf_up_c_cap, 'lf_up_c_ene': lf_up_c_ene,
                'lf_up_d_cap': lf_up_d_cap, 'lf_up_d_ene': lf_up_d_ene,
                'lf_do_c_cap': lf_do_c_cap, 'lf_do_c_ene': lf_do_c_ene,
                'lf_do_d_cap': lf_do_d_cap, 'lf_do_d_ene': lf_do_d_ene}

    @staticmethod
    def add_vars(size):
        """ Adds optimization variables to dictionary

        Variables added:
            lf_up_c (Variable): A cvxpy variable for up power reservation (charge less to the grid)
            lf_do_c (Variable): A cvxpy variable for down power reservation (charge more to the grid)
            lf_up_d (Variable): A cvxpy variable for up power reservation (discharge more to the grid)
            lf_do_d (Variable): A cvxpy variable for down power reservation (discharge less to the grid)

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        # these opt variables are affected by the LF k-up and k-down values to limit the energy throughput level
        return {'lf_up_c': cvx.Variable(shape=size, name='lf_up_c'),
                'lf_do_c': cvx.Variable(shape=size, name='lf_do_c'),
                'lf_up_d': cvx.Variable(shape=size, name='lf_up_d'),
                'lf_do_d': cvx.Variable(shape=size, name='lf_do_d')}

    def objective_constraints(self, variables, mask, load, generation, reservations=None):
        """ Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
                reservations (Dict): power reservations from dispatch services

        Returns:

        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-variables['lf_up_c'])]
        constraint_list += [cvx.NonPos(-variables['lf_do_c'])]
        constraint_list += [cvx.NonPos(-variables['lf_up_d'])]
        constraint_list += [cvx.NonPos(-variables['lf_do_d'])]
        if self.combined_market:
            constraint_list += [cvx.Zero(variables['lf_do_d'] + variables['lf_do_c'] - variables['lf_up_d'] - variables['lf_up_c'])]

        return constraint_list

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

        eta = self.storage.rte

        # calculate reservations
        # c_max, d_min is LF down for charge more, discharge less
        # c_min, d_max is LF up for charge less, discharge more
        c_max = opt_vars['lf_do_c']
        c_min = opt_vars['lf_up_c']
        d_min = opt_vars['lf_do_d']
        d_max = opt_vars['lf_up_d']

        # e_upper = discharge more (LF down) + charge more (LF down) - discharge less (LF up) - charge less (LF up)
        # e (ene throughput) = discharge (LF down) + charge (LF down) - discharge (LF up) - charge (LF up)
        # e_lower = discharge less (LF down) + charge less (LF down) - discharge more (LF up) - charge more (LF up)

        # worst case for upper level of energy throughput
        e_upper = self.kd_max * self.dt * opt_vars['lf_do_d'] + self.kd_max * self.dt * opt_vars['lf_do_c'] * eta - \
                  self.ku_min * self.dt * opt_vars['lf_up_d'] - self.ku_min * self.dt * opt_vars['lf_up_c'] * eta
        # energy throughput is result from combination of ene_throughput for
        # (+ down_discharge + down_charge - up_discharge - up_charge)
        e = cvx.multiply(self.kd * self.dt, opt_vars['lf_do_d']) + \
            cvx.multiply(self.kd * self.dt * eta, opt_vars['lf_do_c']) - \
            cvx.multiply(self.ku * self.dt, opt_vars['lf_up_d']) - \
            cvx.multiply(self.ku * self.dt * eta, opt_vars['lf_up_c'])
        # worst case for lower level of energy throughput
        e_lower = self.kd_min * self.dt * opt_vars['lf_do_d'] + self.kd_min * self.dt * opt_vars['lf_do_c'] * eta - \
                  self.ku_max * self.dt * opt_vars['lf_up_d'] - self.ku_max * self.dt * opt_vars['lf_up_c'] * eta

        # save reservation for optimization window
        self.e.append(e)
        self.e_lower.append(e_lower)
        self.e_upper.append(e_upper)
        self.c_max.append(c_max)
        self.c_min.append(c_min)
        self.d_max.append(d_max)
        self.d_min.append(d_min)
        return [c_max, c_min, d_max, d_min], [e_upper, e, e_lower]

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.price.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                source_data = self.price[self.price.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.price_growth, source_year, yr, frequency)
                self.price = pd.concat([self.price, new_data], sort=True)  # add to existing

                source_data = self.price_lf_up[self.price_lf_up.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.price_growth, source_year, yr, frequency)
                self.price_lf_up = pd.concat([self.price_lf_up, new_data], sort=True)  # add to existing

                source_data = self.price_lf_do[self.price_lf_do.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.price_growth, source_year, yr, frequency)
                self.price_lf_do = pd.concat([self.price_lf_do, new_data], sort=True)  # add to existing

                # Assume that yearly growth rate for k-values is 0 %/year - TN
                source_data = self.ku[self.ku.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, 0, source_year, yr, frequency)
                self.ku = pd.concat([self.ku, new_data], sort=True)  # add to existing

                source_data = self.kd[self.kd.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, 0, source_year, yr, frequency)
                self.kd = pd.concat([self.kd, new_data], sort=True)  # add to existing

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, "LF Energy Throughput (kWh)"] = self.ene_results['ene']
        report.loc[:, "LF Energy Throughput Up (Charging) (kWh)"] = self.ku * self.dt * self.storage.rte * self.variables['lf_up_c']
        report.loc[:, "LF Energy Throughput Up (Discharging) (kWh)"] = self.ku * self.dt * self.variables['lf_up_d']
        report.loc[:, "LF Energy Throughput Down (Charging) (kWh)"] = self.kd * self.dt * self.storage.rte * self.variables['lf_do_c']
        report.loc[:, "LF Energy Throughput Down (Discharging) (kWh)"] = self.kd * self.dt * self.variables['lf_do_d']
        report.loc[:, "LF Energy Settlement Price Signal ($/kWh)"] = self.price
        report.loc[:, 'LF Up (Charging) (kW)'] = self.variables['lf_up_c']
        report.loc[:, 'LF Up (Discharging) (kW)'] = self.variables['lf_up_d']
        report.loc[:, 'LF Up Price Signal ($/kW)'] = self.price_lf_up
        report.loc[:, 'LF Down (Charging) (kW)'] = self.variables['lf_do_c']
        report.loc[:, 'LF Down (Discharging) (kW)'] = self.variables['lf_do_d']
        report.loc[:, 'LF Down Price Signal ($/kW)'] = self.price_lf_do

        return report

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

        lf_up = results.loc[:, 'LF Up (Charging) (kW)'] + results.loc[:, 'LF Up (Discharging) (kW)']
        lf_down = results.loc[:, 'LF Down (Charging) (kW)'] + results.loc[:, 'LF Down (Discharging) (kW)']

        energy_throughput = - results.loc[:, "LF Energy Throughput Down (Discharging) (kWh)"] \
                            - results.loc[:, "LF Energy Throughput Down (Charging) (kWh)"] / self.storage.rte \
                            + results.loc[:, "LF Energy Throughput Up (Discharging) (kWh)"] \
                            + results.loc[:, "LF Energy Throughput Up (Charging) (kWh)"] / self.storage.rte
        energy_through_prof = np.multiply(energy_throughput,
                                          results.loc[:, "LF Energy Settlement Price Signal ($/kWh)"])

        lf_up_prof = np.multiply(lf_up, self.price_lf_up)
        lf_down_prof = np.multiply(lf_down, self.price_lf_do)
        # combine all potential value streams into one df for faster splicing into years
        lf_results = pd.DataFrame({'Energy': energy_through_prof,
                                   'LF up': lf_up_prof,
                                   'LF down': lf_down_prof}, index=results.index)
        for year in opt_years:
            year_monthly = lf_results[lf_results.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), 'LF Energy Throughput'] = year_monthly['Energy'].sum()
            proforma.loc[pd.Period(year=year, freq='y'), 'LF Up'] = year_monthly['LF up'].sum()
            proforma.loc[pd.Period(year=year, freq='y'), 'LF Down'] = year_monthly['LF down'].sum()

        return proforma, None, None

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

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
                self.price_lf_up = np.divide(fr_price, 2)
                self.price_lf_do = np.divide(fr_price, 2)

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
            except KeyError:
                pass
        else:
            try:
                self.price_lf_do = time_series_data.loc[:, 'LF Down Price ($/kW)']
            except KeyError:
                pass

            try:
                self.price_lf_up = time_series_data.loc[:, 'LF Up Price ($/kW)']
            except KeyError:
                pass

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
            except KeyError:
                pass
