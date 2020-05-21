"""
Backup.py

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
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib
import pandas as pd
import numpy as np
import logging

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class Backup(ValueStream):
    """ Backup Power Service. Each service will be daughters of the ValueStream class.

    """

    def __init__(self, params, techs, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            dt (float): optimization timestep (hours)
        """

        # generate the generic service object
        ValueStream.__init__(self, techs['Storage'], 'Backup', dt)
        self.energy_req = params['daily_energy']
        self.monthly_energy = params['monthly_energy']
        self.price = params['monthly_price']
        self.monthly_financial_result = pd.DataFrame()

        # backup energy adds a minimum energy level
        self.ene_min_add = Const.Constraint('ene_min', self.name, self.energy_req)
        self.constraints = {'ene_min': self.ene_min_add}

    def estimate_year_data(self, years, frequency):
        """ Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data

        """
        data_year = self.energy_req.index.year.unique()
        no_data_year = {pd.Period(year) for year in years} - {pd.Period(year) for year in data_year}  # which years do we not have data for

        if len(no_data_year) > 0:
            for yr in no_data_year:
                source_year = pd.Period(max(data_year))
                source_data = self.energy_req[self.energy_req.index.year == source_year.year]  # use source year data
                # growth rate is 0 because source_data is originally given monthly
                new_data = Lib.apply_growth(source_data, 0, source_year, yr, frequency)
                self.energy_req = pd.concat([self.energy_req, new_data], sort=True)  # add to existing

    def monthly_report(self):
        """  Calculates the monthly cost or benefit of the service and adds them to the monthly financial result dataframe

        Returns: A dataframe with the monthly input price of the service and the calculated monthly value in respect
                for each month

        """

        self.monthly_financial_result = pd.DataFrame({'Backup Price ($/kWh)': self.price}, index=self.price.index)
        self.monthly_financial_result.index.names = ['Year-Month']

        return self.monthly_financial_result

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
        proforma, _, _ = ValueStream.proforma_report(self, opt_years, results)
        proforma[self.name] = 0

        # TODO: there should be a check to make sure the commitments where actually met before including it --HN
        for year in opt_years:
            monthly_benefit = np.multiply(self.monthly_energy, self.price)
            proforma.loc[pd.Period(year=year, freq='y')] = monthly_benefit.sum()

        return proforma, [self.name], None

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        report = pd.DataFrame(index=self.energy_req.index)
        report.loc[:, 'Backup Energy Min Added (kWh)'] = self.energy_req
        return report

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not require all
        price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = monthly_data.loc[:, 'Backup Price ($/kWh)']
        except KeyError:
            pass



