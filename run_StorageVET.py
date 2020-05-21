"""
runStorageVET.py

This Python script serves as the initial launch point executing the Python-based version of StorageVET
(AKA StorageVET 2.0 or SVETpy).
"""

__author__ = 'Halley Nathwani, Thien Nguyen, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import argparse
import Scenario
from Params import Params
from Result import Result
import Finances as Fin
import logging
import time
from Visualization import Visualization

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class StorageVET:
    """ StorageVET API. This will eventually allow StorageVET to be imported and used like any
    other python library.

    """

    def __init__(self, model_parameters_path, verbose=False):
        """ Constructor to initialize the parameters and data needed to run StorageVET.
        Initialize the Params Object from Model Parameters

            Args:
                model_parameters_path (str): Filename of the model parameters CSV or XML that
                    describes the case to be analysed
        """
        self.verbose = verbose
        # Initialize the Params Object from Model Parameters
        self.case_dict = Params.initialize(model_parameters_path, verbose)  # unvalidated case instances
        self.results = Result.initialize(Params.results_inputs, Params.case_definitions)
        if verbose:
            self.visualization = Visualization(Params)
            self.visualization.class_summary()

    def solve(self):
        """ Run storageVET

        Returns: the Results class

        """
        starts = time.time()
        if Params.storagevet_requirement_check():
            for key, value in self.case_dict.items():
                run = Scenario.Scenario(value)
                run.add_technology()
                run.add_services()
                run.fill_and_drop_extra_data()
                run.add_control_constraints()
                run.optimize_problem_loop()

                Result.add_instance(key, run)  # cost benefit analysis is in the Result class

            Result.sensitivity_summary()

        ends = time.time()
        print("Full runtime: " + str(ends - starts)) if self.verbose else None
        return Result


if __name__ == '__main__':
    """
        the Main section for runStorageVET to run by itself without the GUI 
    """

    parser = argparse.ArgumentParser(prog='StorageVET.py',
                                     description='The Electric Power Research Institute\'s energy storage system ' +
                                                 'analysis, dispatch, modelling, optimization, and valuation tool' +
                                                 '. Should be used with Python 3.6.x, pandas 0.19+.x, and CVXPY' +
                                                 ' 0.4.x or 1.0.x.',
                                     epilog='Copyright 2018. Electric Power Research Institute (EPRI). ' +
                                            'All Rights Reserved.')
    parser.add_argument('parameters_filename', type=str,
                        help='specify the filename of the CSV file defining the PARAMETERS dataframe')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='specify this flag for verbose output during execution')
    parser.add_argument('--gitlab-ci', action='store_true',
                        help='specify this flag for gitlab-ci testing to skip user input')
    arguments = parser.parse_args()

    case = StorageVET(arguments.parameters_filename, arguments.verbose)
    case.solve()

    # print("Program is done.")
