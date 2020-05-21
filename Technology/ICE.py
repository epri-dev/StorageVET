"""
Diesel

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import cvxpy as cvx
import numpy as np
import pandas as pd
from .DER import DER


class ICE(DER):
    """ An ICE generator

    """

    def __init__(self, name, params):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        DER.__init__(self, params['name'], 'ICE', params)
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.rated_power = params['rated_power']  # kW/generator
        self.p_min = params['min_power']  # kW/generator
        self.startup_time = params['startup_time']  # default value of 0, in units of minutes
        self.efficiency = params['efficiency']  # gal/kWh
        self.fuel_cost = params['fuel_cost']  # $/gal
        self.vari_om = params['variable_om_cost']  # $/kwh
        self.fixed_om = params['fixed_om_cost']  # $/yr
        self.capital_cost = params['ccost']  # $/generator
        self.ccost_kw = params['ccost_kW']

        self.variable_names = {'ice_gen', 'on_ice'}
        try:
            self.n = params['n']  # generators
            self.capex = self.capital_cost * self.n + self.ccost_kw * self.rated_power * self.n
        except KeyError:
            pass

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            ice_gen (Variable): A cvxpy variable equivalent to dis in batteries/CAES
                in terms of ability to provide services
            on_ice (Variable): A cvxpy boolean variable for [...]

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        variables = {'ice_gen': cvx.Variable(shape=size, name='ice_gen', nonneg=True),
                     'on_ice': cvx.Variable(shape=size, boolean=True, name='on_ice')}
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
        ice_gen = variables['ice_gen']
        self.costs = {'ice_fuel': cvx.sum(cvx.multiply(self.efficiency * self.fuel_cost * self.dt * annuity_scalar, variables['ice_gen'])),
                      'ice_fixed': self.fixed_om * annuity_scalar,
                      'ice_variable': cvx.sum(cvx.multiply(self.vari_om * self.dt * annuity_scalar, ice_gen)),
                      'ice_ccost': self.capital_cost * self.n + self.ccost_kw * self.rated_power * self.n
                      }

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
        ice_gen = variables['ice_gen']
        on_ice = variables['on_ice']

        constraint_list += [cvx.NonPos(cvx.multiply(self.p_min, on_ice) - ice_gen)]
        constraint_list += [cvx.NonPos(ice_gen - cvx.multiply(self.rated_power*self.n, on_ice))]

        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        try:
            n = self.n.value
        except AttributeError:
            n = self.n
        results = pd.DataFrame(index=self.variables.index)
        results['ICE Generation (kW)'] = self.variables['ice_gen']
        results['ICE On (y/n)'] = self.variables['on_ice']
        results['ICE P_min (kW)'] = self.p_min
        results['ICE Genset P_max (kW)'] = self.rated_power * n
        return results

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
        pro_forma = DER.proforma_report(self, opt_years, results)
        fuel_col_name = self.name + ' Fuel Costs'
        variable_col_name = self.name + ' Variable O&M Costs'

        ice_gen = self.variables['ice_gen']

        for year in opt_years:
            ice_gen_sub = ice_gen.loc[ice_gen.index.year == year]

            # add variable costs
            pro_forma.loc[pd.Period(year=year, freq='y'), variable_col_name] = -np.sum(self.vari_om * self.dt * ice_gen_sub)

            # add fuel costs
            pro_forma.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(self.efficiency * self.fuel_cost * self.dt * ice_gen_sub)

        return pro_forma
