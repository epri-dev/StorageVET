"""
Constraint.py

This Python class creates a constraint structure within StorageVet.
"""

__author__ = 'Halley Nathwani and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import Library as sh
import logging

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')

class Constraint:
    """ This python class creates a structure that all constraints must follow within StorageVet.

    """

    def __init__(self, name, parent, constraint):
        """Initialize a constraint.

        Args:
            name (str): The name of the constraint--should be "ene_min", "ene_max", "ch_min", "ch_max", "dis_min", or "dis_max".
            parent (str): Indicates what "service" or "technology" generated the constraint
            constraint (float, Series): Constraint value; float if physical constraint, Series with Timestamp index if control or service constraint.

        """
        self.name = name
        self.value = constraint
        # TODO [suggestion] make this the object rather than a string name -MBL???
        self.parent = parent

    def __eq__(self, other):
        """ Determines whether Constraint object equals Constraint case object.

        Args:
            other (Constraint): Constraint object to compare

        Returns:
            bool: True if objects are close to equal, False if not equal.
        """

        return sh.compare_class(self, other)
