"""
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from .DER import DER
import cvxpy as cvx
import pandas as pd
try:
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Library as Lib


class CurtailPV(DER):
    """ Pre_IEEE 1547 2018 standards. Assumes perfect foresight. Ability to curtail PV generation, unlike ChildPV.

    """

    def __init__(self, name, params):
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            name (str): A unique string name for the technology being added, also works as category.
            params (dict): Dict of parameters
        """
        # add startup time parameter (default 0, units minutes)
        self.startup_time = params['startup_time']

        # create generic technology object
        DER.__init__(self, params['name'], 'PV', params)

        self.growth = params['growth']
        self.gen_per_rated = params['rated gen']
        self.rated_capacity = params['rated_capacity']
        self.cost_per_kW = params['cost_per_kW']
        self.loc = params['loc'].lower()
        self.grid_charge = params['grid_charge']
        self.inv_max = params['inv_max']

        self.capex = self.cost_per_kW * self.rated_capacity
        self.generation = self.rated_capacity * self.gen_per_rated

        self.variable_names = {'pv_out'}

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'pv_out': cvx.Variable(shape=size, name='pv_out', nonneg=True)}
        # self.variables = variables
        return variables

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its
            service constraints.
        """
        constraint_list = [cvx.NonPos(variables['pv_out'] - self.generation[mask])]

        if not self.grid_charge:
            constraint_list += [cvx.NonPos(variables['ch'] - variables['pv_out'])]

        if self.loc == 'ac':
            constraint_list += [cvx.NonPos(variables['pv_out'] - self.inv_max)]
            constraint_list += [cvx.NonPos(- self.inv_max - variables['pv_out'])]
        if self.loc == 'dc':
            constraint_list += [cvx.NonPos(variables['pv_out'] + variables['dis'] - variables['ch'] - self.inv_max)]
            constraint_list += [cvx.NonPos(- self.inv_max - variables['pv_out'] - variables['dis'] + variables['ch'])]

        return constraint_list

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.opt_results
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.expressions (Dict): Dict of objective expressions
        """
        self.costs = {'PV capital cost': self.cost_per_kW*self.rated_capacity}
        return self.costs

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.generation.index.year.unique()

        # which years is data given for that is not needed
        dont_need_year = {pd.Period(year) for year in data_year} - {pd.Period(year) for year in years}
        if len(dont_need_year) > 0:
            for yr in dont_need_year:
                generation_sub = self.generation[self.generation.index.year != yr.year]  # choose all data that is not in the unneeded year
                self.generation = generation_sub

        data_year = self.generation.index.year.unique()
        # which years do we not have data for
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}
        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))

                source_data = self.generation[self.generation.index.year == source_year.year]  # use source year data
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.generation = pd.concat([self.generation, new_data], sort=True)  # add to existing

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables.index)
        results['PV Generation (kW)'] = self.variables['pv_out']
        results['PV Maximum (kW)'] = self.generation
        return results
