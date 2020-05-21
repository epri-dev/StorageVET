"""
Technology

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import logging
import pandas as pd
import numpy as np

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class DER:
    """ A general template for Technology object

    We define a "Technology" as anything that might have an effect on the power into or out of the system
    in question.

    """

    def __init__(self, name, category, params):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added
            category (str): A string that represents the type of DER
            params (dict): Dict of parameters
        """

        # initialize internal attributes
        self.name = name
        self.control_constraints = {}
        self.services = {}
        self.predispatch_services = {}
        self.type = category
        self.costs = {}
        self.load = 0
        self.generation = 0
        self.capex = 0  # include in optimization iff sizing
        self.fixed_om = 0  # include in optimization iff sizing
        self.variable_om = 0
        self.operation_cost = 0
        self.dt = params['dt']

        try:
            self.macrs = params['macrs_term']
            self.construction_date = params['construction_date']
            self.operation_date = params['operation_date']
            self.nsr_response_time = params['nsr_response_time']
            self.sr_response_time = params['sr_response_time']
            self.startup_time = params['startup_time']  # startup time, default value of 0, units in minutes
        except KeyError:
            pass

        # attributes about specific to each DER
        self.variables = pd.DataFrame()  # optimization variables are saved here
        self.variable_names = {}  # used to search variable dictionary for DER specific variables
        self.zero_column_name = self.name + ' Capital Cost'  # used for proforma creation
        self.fixed_column_name = self.name + ' Fixed O&M Cost'  # used for proforma creation

    def add_value_streams(self, service, predispatch=False):
        """ Adds a service to the list of services provided by the technology.

        Args:
            service (:obj, ValueStream): A ValueStream class object
            predispatch (Boolean): Flag to add predispatch or dispatch service
        """
        if predispatch:
            self.predispatch_services[service.name] = service
        else:
            self.services[service.name] = service

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        pass

    def calculate_control_constraints(self, datetimes):
        """ Generates a list of master or 'control constraints' from physical constraints and all
        predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.

        Returns:
            Array of datetimes where the control constraints conflict and are infeasible. If all feasible return None.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        """
        return None

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {}

        return variables

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        return self.costs

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
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = []
        return constraint_list

    def update_data(self, fin_inputs):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            fin_inputs (DataFrame): the mutated time_series data with extrapolated price data

        """
        pass

    def save_variable_results(self, variables, subs_index):
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        DER instance and saves the values it to itself

        Args:
            variables (Dict): dictionary of optimization variables after solving the CVXPY problem
            subs_index (Index): index of the subset of data for which the variables were solved for

        """
        variable_values = pd.DataFrame({name: variables[name].value for name in self.variable_names}, index=subs_index)
        self.variables = pd.concat([self.variables, variable_values], sort=True)

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        return None

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
        opt_years = [pd.Period(year=item, freq='y') for item in opt_years]
        pro_forma = pd.DataFrame(data=np.zeros(len(opt_years)), index=opt_years)
        # add capital costs
        pro_forma.columns = [self.zero_column_name]
        try:
            capex = -self.capex.value
        except AttributeError:
            capex = -self.capex
        pro_forma.loc['CAPEX Year', self.zero_column_name] = capex

        pro_forma[self.fixed_column_name] = 0
        for year in opt_years:
            # add fixed o&m costs
            pro_forma.loc[year, self.fixed_column_name] = -self.fixed_om

        return pro_forma
