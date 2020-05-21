"""
Load

This Python class contains methods and attributes specific for technology analysis within DERVET.
"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.0.2'

import numpy as np
import pandas as pd
from Technology.DER import DER


class Load(DER):
    """ An Load object

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        # create generic technology object
        DER.__init__(self, params['name'], 'Load', params)
        self.dt = params['dt']
        self.site_load = params['site_load']

    def get_charge(self, mask):
        return self.site_load[mask].values

    # def get_energy(self, mask):
    #     return np.zeros(len(mask))

    def effective_load(self):
        """ Returns the

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
        """
        return self.site_load.loc[:]

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables.index)
        results["Load (kW)"] = self.effective_load()
        return results
