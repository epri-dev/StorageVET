"""
Copyright (c) 2023, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
SystemRequirement.py

This file hold 2 classes: Requirements and System Requirements
"""

import storagevet.Library as Lib
import numpy as np
import pandas as pd


VERY_LARGE_NUMBER = 2**32 - 1
VERY_LARGE_NEGATIVE_NUMBER = -1 * VERY_LARGE_NUMBER


class Requirement:
    """ This class creates a requirement that needs to be meet (regardless of what other requirements exist)

    """

    def __init__(self, constraint_type, limit_type, parent, constraint):
        """Initialize the requirement and set its value

        Args:
            constraint_type (str): state of energy, charge, or discharge (not case sensitive)
            limit_type (str): maximum limit or minimum (not case sensitive, but must be "max" or "min")
            parent (str): Indicates what "service" or "technology" generated the constraint
            constraint (float, Series): Constraint value; float if physical constraint, Series with Timestamp index if control or service constraint.

        """
        self.type = constraint_type.lower()
        self.limit_type = limit_type.lower()
        self.value = constraint
        self.parent = parent


class SystemRequirement:
    """ This class is meant to handle Requirements of the same type. Determines what min/max value (as a function of time) would
    ensure all of the requirements are met. Hold information about what Value Stream(s) set the value and which others have "contributed"

    """

    def __init__(self, constraint_type, limit_type, years_of_analysis, datetime_freq):
        """Initialize a constraint.

        Args:
            constraint_type (str): state of energy, charge, or discharge (not case sensitive)
            limit_type (str): maximum limit or minimum (not case sensitive)
            years_of_analysis (list): list of years that should be included in the returned Index
            datetime_freq (str): the pandas frequency in string representation -- required to create dateTime rang

        """
        self.type = constraint_type.lower()
        self.is_max = limit_type.lower() == 'max'
        self.is_min = limit_type.lower() == 'min'
        if limit_type not in ['max', 'min']:
            raise SyntaxWarning("limit_type can be 'max' or 'min'")

        index = Lib.create_timeseries_index(years_of_analysis, datetime_freq)
        size = len(index)
        self.parents = pd.DataFrame(columns=['DateTime', 'Parent'])  # records which valuestreams have had a non-base value
        self.owner = pd.Series(np.repeat('null', size), index=index)  # records which valuestream(s) set the current VALUE

        # records the value that would ensure all past updated requirements would also be met
        #   this is needed because of the way that system requirements are created.
        #   you start with all huge or hugely negative values and then values become updated
        #   the update() method is used to build values into these requirements
        #   and lets you handle multiple system requirements of the same type,
        #   without creating a new constraint for each one.
        if self.is_min:
            self.value = pd.Series(np.repeat(VERY_LARGE_NEGATIVE_NUMBER, size), index=index)
        if self.is_max:
            self.value = pd.Series(np.repeat(VERY_LARGE_NUMBER, size), index=index)

    def update(self, requirement):
        """ Update constraint values for the times that were included in. Also update the list tracking which
        value streams are contributing to which timesteps.

        Args:

            requirement (Requirement): the requirement that that VALUE must also be able to meet

        """
        parent = requirement.parent
        value = requirement.value
        update_indx = value.index

        # record the timestamps and parent
        temp_df = pd.DataFrame({'DateTime': update_indx.values})
        temp_df['Parent'] = parent
        self.parents = self.parents.append(temp_df, ignore_index=True)

        # check whether the value needs to be updated, if so--then also update the owner value
        update_values = self.value.loc[update_indx]
        if self.is_min:
            # if minimum constraint, choose higher constraint value
            new_constraint = np.maximum(value.values, update_values.values)
        else:
            # if maximum constraint, choose lower constraint value
            new_constraint = np.minimum(value.values, update_values.values)
        self.value.loc[update_indx] = new_constraint
        # figure out which values changed, and at which indexes
        mask = update_values != new_constraint

        # update the owner at the indexes found above
        self.owner[update_indx][mask] = parent

    def contributors(self, datetime_indx):
        """
        Gets the Parents of the constraint during specified times

        Args:
            datetime_indx (pd.Index): data time index for the timesteps to be considered

        Returns: list of strings that represent the contributors of the constraint value(s) indicated by DATETIME_INDX

        """
        contributors = self.parents[self.parents['DateTime'].isin(datetime_indx.to_list())].Parent
        return contributors.unique()

    def get_subset(self, mask):
        """

        Args:
            mask (DataFrame): DataFrame of booleans used, the same length as time_series. The value is true if the
                corresponding column in time_series is included in the data to be optimized.

        Returns: the value of this requirement at the times that correspond to the mask given

        """
        return self.value.loc[mask].values

    def __le__(self, other):
        """  x<=y calls x.__le__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value <= other.value
        except AttributeError:
            return self.value <= other

    def __lt__(self, other):
        """  x<y calls x.__lt__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value < other.value
        except AttributeError:
            return self.value < other

    def __eq__(self, other):
        """  x==y calls x.__eq__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __ne__(self, other):
        """  x!=y calls x.__ne__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value != other.value
        except AttributeError:
            return self.value != other

    def __gt__(self, other):
        """  x>y calls x.__gt__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value > other.value
        except AttributeError:
            return self.value > other

    def __ge__(self, other):
        """  x>=y calls x.__ge__(y)

        Args:
            other (SystemRequirement, int):

        Returns: bool

        """
        try:
            return self.value >= other.value
        except AttributeError:
            return self.value >= other
