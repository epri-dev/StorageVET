"""
POI.py

"""

__author__ = 'Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'


import xml.etree.ElementTree as et
import pandas as pd
import sys
import ast
import itertools
import logging
import copy
import csv
import Library as sh
import numpy as np
from prettytable import PrettyTable
from datetime import datetime
import collections
import cvxpy as cvx

import matplotlib.pyplot as plt
from pandas import read_csv


class POI:
    """
        This class holds the load data for the case described by the user defined model parameter. It will also
        impose any constraints that should be opposed at the microgrid's POI.
    """

    def __init__(self, params):
        """ Initialize

        Args:
            params (Dict): the dictionary of user inputs
        """

        # coming up with class variables for optimization variables, constraints, and objectives aggregated by all the
        # input technologies

        self.auxiliary_load = params['aux']
        self.site_load = params['site']
        self.system_load = params['system']
        self.deferral_load = params['deferral']
        # self.include_site = params['incl_site']
        # self.include_aux = params['incl_aux']
        self.no_export = params['no_export']
        self.no_import = params['no_import']
        self.default_growth = params['default_growth']
        self.deferral_growth = params['deferral_growth']

    def objective_constraints(self, variables, mask, reservations, generation, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            generation (list, Expression): the sum of generation within the system for the subset of time
                being optimized
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = []
        if self.no_export:
            constraint_list += [cvx.NonPos(variables['dis']-variables['ch']+generation-self.total_load(mask))]
        if self.no_import:
            constraint_list += [cvx.NonPos(-variables['dis']+variables['ch']-generation+self.total_load(mask))]
        return constraint_list
