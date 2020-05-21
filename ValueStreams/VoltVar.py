"""
VoltVar.py

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
import math
import pandas as pd
import logging
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class VoltVar(ValueStream):
    """ Reactive power support, voltage control, power quality. Each service will be daughters of the PreDispService class.
    """

    def __init__(self, params, techs, dt):
        """ Generates the objective function, finds and creates constraints.

          Args:
            params (Dict): input parameters
            techs (Dict): technology objects after initialization, as saved in a dictionary
            dt (float): optimization timestep (hours)
        """

        # generate the generic service object
        ValueStream.__init__(self, techs['Storage'], 'Volt Var', dt)

        # check to see if PV is included and is 'dc' connected to the
        pv_max = 0
        if 'PV' in techs.keys:
            if techs['PV'].loc == 'dc':
                # use inv_max of the inverter shared by pv and ess and save pv generation
                self.inv_max = techs['PV'].inv_max
                pv_max = techs['PV'].generation
        else:
            # otherwise just use the storage's rated discharge
            self.inv_max = techs['Storage'].dis_max_rated

        # add voltage support specific attributes
        self.vars_percent = params['percent']  # TODO: make sure
        self.price = params['price']

        self.vars_reservation = self.vars_percent * self.inv_max

        # # save load
        # self.load = load_data['load']

        # constrain power s.t. enough vars are being outted as well
        power_sqrd = (self.inv_max**2) - (self.vars_reservation**2)

        dis_max = math.sqrt(power_sqrd) - pv_max
        ch_max = math.sqrt(power_sqrd)

        dis_min = 0
        ch_min = 0

        self.constraints = {'ch_max': ch_max,
                            'dis_min': dis_min,
                            'ch_min': ch_min,
                            'dis_max': dis_max}

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
        proforma.columns = [self.name + ' Value']

        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.price

        return proforma, [self.name + ' Value'], None

    def update_yearly_value(self, new_value: float):
        """ Updates the attribute associated to the yearly value of this service. (used by CBA)

        Args:
            new_value (float): the dollar yearly value to be assigned for providing this service

        """
        self.price = new_value
