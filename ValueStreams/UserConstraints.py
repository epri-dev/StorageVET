"""
UserConstraints.py

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
import logging
import pandas as pd
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class UserConstraints(ValueStream):
    """ User entered time series constraints. Each service will be daughters of the PreDispService class.

    """

    def __init__(self, params, tech, dt):
        """ Generates the objective function, finds and creates constraints.

        Acceptable constraint names are: 'Power Max (kW)', 'Power Min (kW)', 'Energy Max (kWh)', 'Energy Min (kWh)'

          Args:
            params (Dict): input parameters
            tech (Technology): Storage technology object
            dt (float): optimization timestep (hours)
        """
        # generate the generic service object
        ValueStream.__init__(self, tech, 'User Constraints', dt)

        self.user_constraints = params['constraints']
        self.price = params['price']  # $/yr

        # save original values to check for infeasibility
        power_max, power_min = 0, 0
        energy_max, energy_min = 0, 0

        # look at only the columns in input_cols
        if self.user_constraints.columns.isin(['Power Max (kW)']).any():
            power_max = self.user_constraints['Power Max (kW)']
            self.constraints['dis_max'] = Const.Constraint('dis_max', self.name, power_max.clip(lower=0))
            self.constraints['ch_min'] = Const.Constraint('ch_min', self.name, -power_max.clip(upper=0))

        if self.user_constraints.columns.isin(['Power Min (kW)']).any():
            power_min = self.user_constraints['Power Min (kW)']
            self.constraints['dis_min'] = Const.Constraint('dis_min', self.name, power_min.clip(lower=0))
            self.constraints['ch_max'] = Const.Constraint('ch_max', self.name, -power_min.clip(upper=0))

        if self.user_constraints.columns.isin(['Energy Max (kWh)']).any():
            energy_max = self.user_constraints['Energy Max (kWh)']
            self.constraints['ene_max'] = Const.Constraint('ene_max', self.name, energy_max)

        if self.user_constraints.columns.isin(['Energy Min (kWh)']).any():
            energy_min = self.user_constraints['Energy Min (kWh)']
            self.constraints['ene_min'] = Const.Constraint('ene_min', self.name, energy_min)

        # if both a power max and min are given, then preform this check to check for infeasibility
        if type(power_max) == pd.Series and type(power_min) == pd.Series:
            if (power_min > power_max).any():
                raise Exception('User given power constraints are not feasible.')

        # if both a energy max and min are given, then preform this check to check for infeasibility
        if type(energy_min) == pd.Series and type(energy_max) == pd.Series:
            if (energy_min > energy_max).any():
                raise Exception('User given energy constraints are not feasible.')

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
        proforma[self.name + ' Value'] = 0

        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.price

        return proforma, [self.name + ' Value'], None

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value
