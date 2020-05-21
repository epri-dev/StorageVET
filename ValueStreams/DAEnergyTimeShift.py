"""
DAEnergyTimeShift.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""

__author__ = 'Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from ValueStreams.ValueStream import ValueStream
import cvxpy as cvx
import pandas as pd
import Library as Lib
import logging
import numpy as np

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class DAEnergyTimeShift(ValueStream):
    """ Day-Ahead Energy Time Shift. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        ValueStream.__init__(self, tech, 'DA ETS', dt)
        self.price = params['price']
        self.growth = params['growth']  # growth rate of energy prices (%/yr)

    def objective_function(self, variables, mask, load, generation, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            variables (Dict): dictionary of variables being optimized
            subs (DataFrame): table of load data for the optimization windows
            generation (list, Expression): the sum of generation within the system
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            The expression of the objective function that it affects. This can be passed into the cvxpy solver.

        """

        size = sum(mask)

        p_da = cvx.Parameter(size, value=self.price.loc[mask].values, name='da price')
        # self.costs.append(temp_exp)
        return {self.name: cvx.sum(-p_da * variables['dis'] + p_da * variables['ch'] - p_da * generation) * annuity_scalar * self.dt}  #

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
                new_data = Lib.apply_growth(source_data, self.growth, source_year, yr, frequency)
                self.price = pd.concat([self.price, new_data], sort=True)  # add to existing

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, 'DA Price Signal ($/kWh)'] = self.price
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

        energy_cost = self.dt * np.multiply(results['Total Generation (kW)'] + results['Total Storage Power (kW)'], self.price)
        for year in opt_years:
            year_subset = energy_cost[energy_cost.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), 'DA ETS'] = year_subset.sum()

        return proforma, None, None

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = time_series_data.loc[:, 'DA Price ($/kWh)']
        except KeyError:
            pass

