"""
FrequencyRegulation.py

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


class FrequencyRegulation(ValueStream):
    """ Frequency Regulation. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.
        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)

        To Do:
            - determine a method to provide values for the min/max values of k-values if min/max are not provided
        """
        # financials_df = financials.fin_inputs
        ValueStream.__init__(self, tech, 'FR', dt)
        # self.fr_energyprice = params['energyprice']
        self.krd_avg = params['kd']
        self.kru_avg = params['ku']
        self.combined_market = params['CombinedMarket']  # boolean: true if storage bid as much reg up as reg down
        self.price = params['energy_price']  # TODO: require RT market price instead of DA
        self.p_regu = params['regu_price']
        self.p_regd = params['regd_price']
        self.growth = params['growth']
        self.energy_growth = params['energyprice_growth']
        self.duration = params['duration']

        self.variable_names = {'regu_c', 'regd_c', 'regu_d', 'regd_d'}
        self.variables = pd.DataFrame(columns=self.variable_names)
        # regulation up due to charging, regulation down due to charging, regulation up due to discharging, regulation down due to discharging

    def objective_function(self, variables, mask, load, generation, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            The portion of the objective function that it affects. This can be passed into the cvxpy solver. Returns costs - benefits

        """
        size = sum(mask)
        # pay for reg down energy, get paid for reg up energy
        # paid revenue for capacity to do both

        p_regu = cvx.Parameter(size, value=self.p_regu.loc[mask].values, name='p_regu')
        p_regd = cvx.Parameter(size, value=self.p_regd.loc[mask].values, name='p_regd')
        p_ene = cvx.Parameter(size, value=self.price.loc[mask].values, name='price')

        regup_charge_payment = cvx.sum(variables['regu_c'] * -p_regu) * annuity_scalar
        regup_charge_settlement = cvx.sum(variables['regu_c'] * -p_ene) * self.dt * self.kru_avg * annuity_scalar

        regup_disch_payment = cvx.sum(variables['regu_d'] * -p_regu) * annuity_scalar
        regup_disch_settlement = cvx.sum(variables['regu_d'] * -p_ene) * self.dt * self.kru_avg * annuity_scalar

        regdown_charge_payment = cvx.sum(variables['regd_c'] * -p_regd) * annuity_scalar
        regdown_charge_settlement = cvx.sum(variables['regd_c'] * p_ene) * self.dt * self.krd_avg * annuity_scalar

        regdown_disch_payment = cvx.sum(variables['regd_d'] * -p_regd) * annuity_scalar
        regdown_disch_settlement = cvx.sum(variables['regd_d'] * p_ene) * self.dt * self.krd_avg * annuity_scalar

        return {'regup_payment': regup_charge_payment + regup_disch_payment,
                'regdown_payment': regdown_charge_payment + regdown_disch_payment,
                'fr_energy_settlement': regup_disch_settlement + regdown_disch_settlement + regup_charge_settlement + regdown_charge_settlement}

    @staticmethod
    def add_vars(size):
        """ Adds optimization variables to dictionary

        Variables added:
            regu_c (Variable): A cvxpy variable for freq regulation capacity to increase charging power
            regd_c (Variable): A cvxpy variable for freq regulation capacity to decrease charging power
            regu_d (Variable): A cvxpy variable for freq regulation capacity to increase discharging power
            regd_d (Variable): A cvxpy variable for freq regulation capacity to decrease discharging power

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        return {'regu_c': cvx.Variable(shape=size, name='regu_c'),
                'regd_c': cvx.Variable(shape=size, name='regd_c'),
                'regu_d': cvx.Variable(shape=size, name='regu_d'),
                'regd_d': cvx.Variable(shape=size, name='regd_d')}

    def objective_constraints(self, variables, mask, load, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
            reservations (Dict): power reservations from dispatch services

        Returns:
            constraint_list (list): list of constraints

        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-variables['regu_c'])]
        constraint_list += [cvx.NonPos(-variables['regd_c'])]
        constraint_list += [cvx.NonPos(-variables['regu_d'])]
        constraint_list += [cvx.NonPos(-variables['regd_d'])]
        # p = opt_vars['dis'] - opt_vars['ch']
        # constraint_list += [cvx.NonPos(opt_vars['regd_d'] - cvx.pos(p))]
        # constraint_list += [cvx.NonPos(opt_vars['regu_c'] - cvx.neg(p))]
        if self.combined_market:
            constraint_list += [cvx.Zero(variables['regd_d'] + variables['regd_c'] - variables['regu_d'] - variables['regu_c'])]

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
        size = opt_vars['ene'].shape

        # calculate reservations
        c_max = opt_vars['regd_c']
        c_min = opt_vars['regu_c']
        d_min = opt_vars['regd_d']
        d_max = opt_vars['regu_d']

        # upper and lower level of energy reservation is now based on the provided duration input of the dispatch service
        # worst case for upper level of energy throughput
        e_upper = opt_vars['regd_c'] * eta * self.duration + opt_vars['regd_d'] * self.duration
        # energy throughput is result from combination of ene_throughput for
        # (+ down_discharge + down_charge - up_discharge - up_charge)
        # double check on the signs
        e = self.krd_avg*self.dt*opt_vars['regd_d'] + self.krd_avg*self.dt*opt_vars['regd_c']*eta - self.kru_avg*self.dt*opt_vars['regu_d'] - self.kru_avg*self.dt*opt_vars['regu_c']*eta
        # worst case for lower level of energy throughput
        e_lower = opt_vars['regu_c'] * eta * self.duration + opt_vars['regu_d'] * self.duration

        # save reservation for optmization window
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
                new_data = Lib.apply_growth(source_data, self.energy_growth, source_year, yr, frequency)
                self.price = pd.concat([self.price, new_data], sort=True)  # add to existing

                source_data = self.p_regu[self.p_regu.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.p_regu = pd.concat([self.p_regu, new_data], sort=True)  # add to existing

                source_data = self.p_regd[self.p_regd.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.p_regd = pd.concat([self.p_regd, new_data], sort=True)  # add to existing

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, "FR Energy Throughput (kWh)"] = self.ene_results['ene']
        report.loc[:, "FR Energy Throughput Up (Charging) (kWh)"] = self.variables['regu_c']*self.krd_avg*self.dt*self.storage.rte
        report.loc[:, "FR Energy Throughput Up (Discharging) (kWh)"] = self.variables['regu_d']*self.krd_avg*self.dt
        report.loc[:, "FR Energy Throughput Down (Charging) (kWh)"] = self.variables['regd_c']*self.krd_avg*self.dt*self.storage.rte
        report.loc[:, "FR Energy Throughput Down (Discharging) (kWh)"] = self.variables['regd_d']*self.krd_avg*self.dt
        report.loc[:, "FR Energy Settlement Price Signal ($/kWh)"] = self.price
        report.loc[:, 'Regulation Up (Charging) (kW)'] = self.variables['regu_c']
        report.loc[:, 'Regulation Up (Discharging) (kW)'] = self.variables['regu_d']
        report.loc[:, 'Regulation Down (Charging) (kW)'] = self.variables['regd_c']
        report.loc[:, 'Regulation Down (Discharging) (kW)'] = self.variables['regd_d']
        report.loc[:, "Regulation Up Price Signal ($/kW)"] = self.p_regu
        report.loc[:, "Regulation Down Price Signal ($/kW)"] = self.p_regd

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

        reg_up = results.loc[:, 'Regulation Up (Charging) (kW)'] + results.loc[:, 'Regulation Up (Discharging) (kW)']
        reg_down = results.loc[:, 'Regulation Down (Charging) (kW)'] + results.loc[:, 'Regulation Down (Discharging) (kW)']

        energy_throughput = - results.loc[:, "FR Energy Throughput Down (Discharging) (kWh)"] \
                            - results.loc[:, "FR Energy Throughput Down (Charging) (kWh)"] / self.storage.rte \
                            + results.loc[:, "FR Energy Throughput Up (Discharging) (kWh)"] \
                            + results.loc[:, "FR Energy Throughput Up (Charging) (kWh)"] / self.storage.rte
        energy_through_prof = np.multiply(energy_throughput, results.loc[:, "FR Energy Settlement Price Signal ($/kWh)"])

        regulation_up_prof = np.multiply(reg_up, self.p_regu)
        regulation_down_prof = np.multiply(reg_down, self.p_regd)
        # combine all potential value streams into one df for faster splicing into years
        fr_results = pd.DataFrame({'Energy': energy_through_prof,
                                   'Reg up': regulation_up_prof,
                                   'Reg down': regulation_down_prof}, index=results.index)
        for year in opt_years:
            year_subset = fr_results[fr_results.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), 'FR Energy Throughput'] = year_subset['Energy'].sum()
            proforma.loc[pd.Period(year=year, freq='y'), 'Regulation Up'] = year_subset['Reg up'].sum()
            proforma.loc[pd.Period(year=year, freq='y'), 'Regulation Down'] = year_subset['Reg down'].sum()

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
                fr_price = time_series_data.loc[:, 'FR Price ($/kW)']
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
                self.p_regu = time_series_data.loc[:, 'Reg Up Price ($/kW)']
            except KeyError:
                pass

            try:
                self.p_regd = time_series_data.loc[:, 'Reg Down Price ($/kW)']
            except KeyError:
                pass

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
            except KeyError:
                pass
