"""
ValueStreamream.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import numpy as np
import cvxpy as cvx
import pandas as pd
import logging
try:
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class ValueStream:
    """ A general template for services provided and constrained by the providing technologies.

    """

    def __init__(self, storage, name, dt):
        """ Initialize all services with the following attributes

        Args:
            storage (Technology): Storage technology object
            name (str): A string name/description for the service
            dt (float): optimization timestep (hours)
        """
        self.name = name

        # power and energy requirements
        self.c_max = []
        self.c_min = []
        self.d_min = []
        self.d_max = []
        self.e_upper = []
        self.e = []
        self.e_lower = []

        self.constraints = {}  # TODO: rename this (used in calculate_control_constraints)

        self.storage = storage
        # TODO: might be cleaner to pass in technologies when there is more than 1 (instead of saving as attributes

        self.dt = dt
        self.ene_results = pd.DataFrame()
        self.variables = pd.DataFrame()  # optimization variables are saved here
        self.variable_names = {}  # used to search variable dictionary for service specific variables
        # self.zero_column_name = self.name + ' Capital Cost'  # used for proforma creation
        # self.fixed_column_name = self.name + ' Fixed O&M Cost'  # used for proforma creation

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
            A dictionary with the portion of the objective function that it affects, labeled by the expression's key. Default is to return {}.
        """
        return {}

    def objective_constraints(self, variables, mask, load, generation, reservations=None):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set            load (list, Expression): the sum of load within the system
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
            reservations (Dict): power reservations from dispatch services

        Returns:
            An empty list
        """
        return []

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
        size = opt_vars['ene'].shape
        # calculate reservations
        c_max = 0
        c_min = 0
        d_min = 0
        d_max = 0
        e_upper = cvx.Parameter(shape=size, value=np.zeros(size), name='e_upper zero')
        e = cvx.Parameter(shape=size, value=np.zeros(size), name='e zero')
        e_lower = cvx.Parameter(shape=size, value=np.zeros(size), name='e_lower zero')

        # save reservation for optmization window
        self.e.append(e)
        self.e_lower.append(e_lower)
        self.e_upper.append(e_upper)
        self.c_max.append(c_max)
        self.c_min.append(c_min)
        self.d_max.append(d_max)
        self.d_min.append(d_min)
        return [c_max, c_min, d_max, d_min], [e_upper, e, e_lower]

    @staticmethod
    def add_vars(size):
        """ Default method that will not create any new optimization variables

        Args:
            size (int): length of optimization variables to create

        Returns:
            An empty set
        """
        return {}

    def save_variable_results(self, variables, subs_index):
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        ValueStream instance and saves the values it to itself

        Args:
            variables (Dict): dictionary of optimization variables after solving the CVXPY problem
            subs_index (Index): index of the subset of data for which the variables were solved for

        """
        variable_values = pd.DataFrame({name: variables[name].value for name in self.variable_names}, index=subs_index)
        self.variables = pd.concat([self.variables, variable_values], sort=True)

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        pass

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        return None

    def monthly_report(self):
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """
        return None

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
        opt_years = [pd.Period(year=item, freq='y') for item in opt_years]
        proforma = pd.DataFrame(index=opt_years)

        return proforma, None, None

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        pass

    # def update_tariff_rate(self, tariff, tariff_price_signal):
    #     """ Updates attributes related to the tariff rate with the provided tariff rate.
    #
    #     Args:
    #         tariff (DataFrame): raw tariff file (as read directly from the user given CSV)
    #         tariff_price_signal (DataFrame): time series form of the tariff -- contains the price for energy
    #             and the billing period(s) that time-step applies to
    #
    #     """
    #     pass
