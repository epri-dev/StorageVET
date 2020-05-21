"""
CAESTech.py

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
import cvxpy as cvx
import logging
import pandas as pd
import numpy as np

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class CAESTech(Storage):
    """ CAES class that inherits from Storage.

    """

    def __init__(self, name,  params):
        """ Initializes a CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            name (string): name of technology
            params (dict): params dictionary from dataframe for one case
        """

        # create generic storage object  TODO: name and category are the same...did we do this on purpose?
        Storage.__init__(self, params['name'], params)
        # add CAES specific attributes
        self.heat_rate_high = params['heat_rate_high']
        self.natural_gas_price = params['natural_gas_price']  # $/MillionBTU

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective costs for fuel cost and O&M cost

         Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs

        """
        # get generic Tech objective costs
        Storage.objective_function(self, variables, mask, annuity_scalar)

        # add fuel cost expression
        fuel_exp = cvx.sum(cvx.multiply(self.natural_gas_price[mask].values, variables['dis']) * self.heat_rate_high
                                                                                    * self.dt * 1e-6 * annuity_scalar)
        self.costs.update({'CAES_fuel_cost': fuel_exp})

        return self.costs

    def calc_operating_cost(self, energy_rate, fuel_rate):
        """ Calculates operating cost in dollars per MWh_out

         Args:
            energy_rate (float): energy rate [=] $/kWh
            fuel_rate (float): natural gas price rate [=] $/MillionBTU

        Returns:
            Value of Operating Cost [=] $/MWh_out

        """
        fuel_cost = fuel_rate*self.heat_rate_high*1e3/1e6
        om = self.fixedOM
        energy = energy_rate*1e3/self.rte

        return fuel_cost + om + energy

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables.index)
        results[self.name + ' Discharge (kW)'] = self.variables['dis']
        results[self.name + ' Charge (kW)'] = self.variables['ch']
        results[self.name + ' Power (kW)'] = self.variables['dis'] - self.variables['ch']
        results[self.name + ' State of Energy (kWh)'] = self.variables['ene']

        try:
            energy_rate = self.ene_max_rated.value
        except AttributeError:
            energy_rate = self.ene_max_rated

        results[self.name + ' SOC (%)'] = self.variables['ene'] / energy_rate
        results[self.name + ' Natural Gas Price ($)'] = self.natural_gas_price

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
        pro_forma = Storage.proforma_report(self, opt_years, results)
        fuel_col_name = self.name + ' Natural Gas Costs'

        for year in opt_years:
            fuel_price_sub = self.natural_gas_price.loc[self.natural_gas_price.index.year == year]
            pro_forma.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(fuel_price_sub*self.heat_rate_high*1e3/1e6)

        return pro_forma
