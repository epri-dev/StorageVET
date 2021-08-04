"""
Copyright (c) 2021, Electric Power Research Institute

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
Technology

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import pandas as pd
import numpy as np
import cvxpy as cvx
from storagevet.ErrorHandling import *


class DER:
    """ A general template for DER object, which could be any kind of Distributed Energy Resources currently
        supported in DERVET: storage (CAES, Battery), generator (CHP, ICE), renewable (PV), and loads


    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        # initialize internal attributes
        self.name = params['name']  # specific tech model name
        self.dt = params['dt']
        self.technology_type = None  # "Energy Storage System", "Rotating Generator", "Intermittent Resource", "Load"
        self.tag = None
        self.variable_om = 0  # $/kWh
        self.id = params.get('ID')

        # attributes about specific to each DER
        self.variables_df = pd.DataFrame()  # optimization variables are saved here
        self.variables_dict = {}  # holds the CVXPY variables upon creation in the technology instance

        # boolean attributes
        self.is_electric = False    # can this DER consume or generate electric power?
        self.is_hot = False         # can this DER consume or generate heat?
        self.is_cold = False        # can this DER consume or generate cooling power?
        self.is_fuel = False        # can this DER consume fuel?

        self.can_participate_in_market_services = True

    def zero_column_name(self):
        return self.unique_tech_id() + ' Capital Cost'  # used for proforma creation

    def fixed_column_name(self):
        return self.unique_tech_id() + ' Fixed O&M Cost'  # used for proforma creation

    def get_capex(self, **kwargs) -> cvx.Variable or float:
        """

        Returns: the capex of this DER

        """
        return 0

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        pass

    def discharge_capacity(self):
        """

        Returns: the maximum discharge that can be attained

        """
        return 0

    def charge_capacity(self):
        """

        Returns: the maximum charge that can be attained

        """
        return 0

    def operational_max_energy(self):
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """

        return 0

    def operational_min_energy(self):
        """

        Returns: the minimum energy that should stored in this DER based on user inputs
        """

        return 0

    def qualifying_capacity(self, event_length):
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        return 0

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:

        Args:
            size (Int): Length of optimization variables to create

        """
        pass

    def get_state_of_energy(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')

    def get_discharge(self, mask):
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')

    def get_net_power(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the net power [= charge - discharge] as a function of time for the

        """
        return self.get_charge(mask) - self.get_discharge(mask)

    def get_charge_up_schedule(self, mask):
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def get_charge_down_schedule(self, mask):
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def get_discharge_up_schedule(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def get_discharge_down_schedule(self, mask):
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def get_delta_uenegy(self, mask):
        """ the amount of energy, from the current SOE level the DER's state of energy changes
        from subtimestep energy shifting

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def get_uenergy_increase(self, mask):
        """ the amount of energy in a timestep that is provided to the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def get_uenergy_decrease(self, mask):
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        Returns:
            costs - benefits (Dict): Dict of objective costs
        """
        return {}

    def constraints(self, mask, **kwargs):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        return []

    def save_variable_results(self, subs_index):
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        DER instance and saves the values it to itself

        Args:
            subs_index (Index): index of the subset of data for which the variables were solved for

        """
        variable_values = pd.DataFrame({name: variable.value for name, variable in self.variables_dict.items()}, index=subs_index)
        self.variables_df = pd.concat([self.variables_df, variable_values], sort=True)

    def unique_tech_id(self):
        """ String id that serves as the prefix for reporting optimization variables for specific DER via timeseries
            or proforma method. USED IN REPORTING ONLY
        """
        return f'{self.tag.upper()}: {self.name}'

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        return pd.DataFrame()

    def monthly_report(self):
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        return {}

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        if not self.zero_column_name():
            return None

        pro_forma = pd.DataFrame({self.zero_column_name(): -self.get_capex(solution=True)}, index=['CAPEX Year'])

        return pro_forma
