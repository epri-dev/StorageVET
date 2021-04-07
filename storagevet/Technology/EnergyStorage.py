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
Storage

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER
from storagevet.ErrorHandling import *


class EnergyStorage(DER):
    """ A general template for storage object

    We define "storage" as anything that can affect the quantity of load/power being delivered or used. Specific
    types of storage are subclasses. The storage subclass should be called. The storage class should never
    be called directly.

    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        super().__init__(params)
        # input params
        # note: these should never be changed in simulation (i.e from degradation)
        self.technology_type = 'Energy Storage System'
        try:
          self.rte = params['rte']/100
        except KeyError:
          self.rte = 1/params['energy_ratio']
        self.sdr = params['sdr']/100
        self.ene_max_rated = params['ene_max_rated']
        self.dis_max_rated = params['dis_max_rated']
        self.dis_min_rated = params['dis_min_rated']
        self.ch_max_rated = params['ch_max_rated']
        self.ch_min_rated = params['ch_min_rated']
        self.ulsoc = params['ulsoc']/100
        self.llsoc = params['llsoc']/100
        self.soc_target = params['soc_target']/100

        self.fixedOM_perKW = params['fixedOM']  # $/kW
        self.variable_om = params['OMexpenses']*1e-3  # $/MWh * 1e-3 = $/kWh
        self.incl_startup = params['startup']
        self.incl_binary = params['binary']
        self.daily_cycle_limit = params['daily_cycle_limit']

        self.capital_cost_function = [params['ccost'], params['ccost_kw'], params['ccost_kwh']]

        if self.incl_startup:
            self.p_start_ch = params['p_start_ch']
            self.p_start_dis = params['p_start_dis']

        # to be changed and reset everytime the effective energy capacity changes
        self.effective_soe_min = self.llsoc * self.ene_max_rated
        self.effective_soe_max = self.ulsoc * self.ene_max_rated

        self.variable_names = {'ene', 'dis', 'ch', 'uene', 'uch', 'udis'}

    def discharge_capacity(self):
        """

        Returns: the maximum discharge that can be attained

        """
        return self.dis_max_rated

    def charge_capacity(self):
        """

        Returns: the maximum charge that can be attained

        """
        return self.ch_max_rated

    def energy_capacity(self, solution=False):
        """

        Returns: the maximum energy that can be attained

        """
        return self.ene_max_rated

    def operational_max_energy(self):
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """

        return self.effective_soe_max

    def operational_min_energy(self):
        """

        Returns: the minimum energy that should stored in this DER based on user inputs
        """

        return self.effective_soe_min

    def qualifying_capacity(self, event_length):
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        return min(self.discharge_capacity(), self.operational_max_energy()/event_length)

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added: (with self.unique_ess_id as a prefix to these)
            ene (Variable): A cvxpy variable for Energy at the end of the time step (kWh)
            dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step (kW)
            ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step (kW)
            on_c (Variable/Parameter): A cvxpy variable/parameter to flag for charging in previous interval (bool)
            on_d (Variable/Parameter): A cvxpy variable/parameter to flag for discharging in previous interval (bool)
            start_c (Variable):  A cvxpy variable to flag to capture which intervals charging started (bool)
            start_d (Variable): A cvxvy variable to flag to capture which intervals discharging started (bool)

        Notes:
            CVX Parameters turn into Variable when the condition to include them is active

        Args:
            size (Int): Length of optimization variables to create

        """
        self.variables_dict = {
            'ene': cvx.Variable(shape=size, name=self.name + '-ene'),
            'dis': cvx.Variable(shape=size, name=self.name + '-dis'),
            'ch': cvx.Variable(shape=size, name=self.name + '-ch'),
            'uene': cvx.Variable(shape=size, name=self.name + '-uene'),
            'udis': cvx.Variable(shape=size, name=self.name + '-udis'),
            'uch': cvx.Variable(shape=size, name=self.name + '-uch'),
            'on_c': cvx.Parameter(shape=size, name=self.name + '-on_c', value=np.ones(size)),
            'on_d': cvx.Parameter(shape=size, name=self.name + '-on_d', value=np.ones(size)),
            'start_c': cvx.Parameter(shape=size, name=self.name + '-start_c', value=np.ones(size)),
            'start_d': cvx.Parameter(shape=size, name=self.name + '-start_d', value=np.ones(size)),
        }

        if self.incl_binary:
            self.variable_names.update(['on_c', 'on_d'])
            self.variables_dict.update({'on_c': cvx.Variable(shape=size, boolean=True, name=self.name + '-on_c'),
                                        'on_d': cvx.Variable(shape=size, boolean=True, name=self.name + '-on_d')})
            if self.incl_startup:
                self.variable_names.update(['start_c', 'start_d'])
                self.variables_dict.update({'start_c': cvx.Variable(shape=size, name=self.name + '-start_c'),
                                            'start_d': cvx.Variable(shape=size, name=self.name + '-start_d')})

    def get_state_of_energy(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        return self.variables_dict['ene']

    def get_discharge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return self.variables_dict['dis']

    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return self.variables_dict['ch']

    def get_capex(self, **kwargs):
        """ Returns the capex of a given technology
        """
        return np.dot(self.capital_cost_function, [1, self.dis_max_rated, self.ene_max_rated])

    def get_fixed_om(self):
        """ Returns the fixed om of a given technology
        """
        return self.fixedOM_perKW * self.dis_max_rated

    def get_charge_up_schedule(self, mask):
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['ch'] - self.ch_min_rated

    def get_charge_down_schedule(self, mask):
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.ch_max_rated - self.variables_dict['ch']

    def get_discharge_up_schedule(self, mask):
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.dis_max_rated - self.variables_dict['dis']

    def get_discharge_down_schedule(self, mask):
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['dis'] - self.dis_min_rated

    def get_delta_uenegy(self, mask):
        """ the amount of energy, from the current SOE level the DER's state of energy changes
        from subtimestep energy shifting

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['uene']

    def get_uenergy_increase(self, mask):
        """ the amount of energy in a timestep that is provided to the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['uch'] * self.dt

    def get_uenergy_decrease(self, mask):
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.variables_dict['udis'] * self.dt

    def objective_function(self, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else annuity_scalar should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """

        # create objective expression for variable om based on discharge activity
        var_om = cvx.sum(self.variables_dict['dis'] + self.variables_dict['udis']) * self.variable_om * self.dt * annuity_scalar

        costs = {
            self.name + ' fixed_om': self.get_fixed_om() * annuity_scalar,
            self.name + ' var_om': var_om
        }
        # add startup objective costs
        if self.incl_startup:
            costs.update({
                self.name + ' ch_startup': cvx.sum(self.variables_dict['start_c']) * self.p_start_ch * annuity_scalar,
                self.name + ' dis_startup': cvx.sum(self.variables_dict['start_d']) * self.p_start_dis * annuity_scalar})

        return costs

    def constraints(self, mask, sizing_for_rel=False, find_min_soe=False):
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        size = int(np.sum(mask))

        ene_target = self.soc_target * self.effective_soe_max   # this is init_ene

        # optimization variables
        ene = self.variables_dict['ene']
        dis = self.variables_dict['dis']
        ch = self.variables_dict['ch']
        uene = self.variables_dict['uene']
        udis = self.variables_dict['udis']
        uch = self.variables_dict['uch']
        on_c = self.variables_dict['on_c']
        on_d = self.variables_dict['on_d']
        start_c = self.variables_dict['start_c']
        start_d = self.variables_dict['start_d']

        if sizing_for_rel:
            constraint_list += [
                cvx.Zero(ene[0] - ene_target + (self.dt * dis[0]) - (self.rte * self.dt * ch[0]) - uene[0] + (ene[0] * self.sdr * 0.01))]
            constraint_list += [
                cvx.Zero(ene[1:] - ene[:-1] + (self.dt * dis[1:]) - (self.rte * self.dt * ch[1:]) - uene[1:] + (ene[1:] * self.sdr * 0.01))]
        else:
            # energy at beginning of time step must be the target energy value
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
            # energy evolution generally for every time step
            constraint_list += [
                cvx.Zero(ene[1:] - ene[:-1] + (self.dt * dis[:-1]) - (self.rte * self.dt * ch[:-1]) - uene[:-1] + (ene[:-1] * self.sdr * 0.01))]

            # energy at the end of the last time step (makes sure that the end of the last time step is ENE_TARGET
            constraint_list += [cvx.Zero(ene_target - ene[-1] + (self.dt * dis[-1]) - (self.rte * self.dt * ch[-1]) - uene[-1] + (ene[-1] * self.sdr * 0.01))]

        # constraints on the ch/dis power
        constraint_list += [cvx.NonPos(ch - (on_c * self.ch_max_rated))]
        constraint_list += [cvx.NonPos((on_c * self.ch_min_rated) - ch)]
        constraint_list += [cvx.NonPos(dis - (on_d * self.dis_max_rated))]
        constraint_list += [cvx.NonPos((on_d * self.dis_min_rated) - dis)]

        # constraints on the state of energy
        constraint_list += [cvx.NonPos(self.effective_soe_min - ene)]
        constraint_list += [cvx.NonPos(ene - self.effective_soe_max)]

        # account for -/+ sub-dt energy -- this is the change in energy that the battery experiences as a result of energy option
        # if sizing for reliability
        if sizing_for_rel:
            constraint_list += [cvx.Zero(uene)]
        else:
            constraint_list += [cvx.Zero(uene + (self.dt * udis) - (self.dt * uch * self.rte))]

        # the constraint below limits energy throughput and total discharge to less than or equal to
        # (number of cycles * energy capacity) per day, for technology warranty purposes
        # this constraint only applies when optimization window is equal to or greater than 24 hours
        if self.daily_cycle_limit and size >= 24:
            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                constraint_list += [cvx.NonPos(cvx.sum(dis[day_mask] + udis[day_mask]) * self.dt - self.ene_max_rated * self.daily_cycle_limit)]
        elif self.daily_cycle_limit and size < 24:
            TellUser.info('Daily cycle limit did not apply as optimization window is less than 24 hours.')

        # note: cannot operate startup without binary
        if self.incl_startup and self.incl_binary:
            # startup variables are positive
            constraint_list += [cvx.NonPos(-start_c)]
            constraint_list += [cvx.NonPos(-start_d)]
            # difference between binary variables determine if started up in
            # previous interval
            constraint_list += [cvx.NonPos(cvx.diff(on_d) - start_d[1:])]
            constraint_list += [cvx.NonPos(cvx.diff(on_c) - start_c[1:])]
        return constraint_list

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = super().timeseries_report()
        solve_dispatch_opt = self.variables_df.get('dis')
        if solve_dispatch_opt is not None:
            results[tech_id + ' Discharge (kW)'] = self.variables_df['dis']
            results[tech_id + ' Charge (kW)'] = -self.variables_df['ch']
            results[tech_id + ' Power (kW)'] = \
                self.variables_df['dis'] - self.variables_df['ch']
            results[tech_id + ' State of Energy (kWh)'] = \
                self.variables_df['ene']

            results[tech_id + ' Energy Option (kWh)'] =\
                self.variables_df['uene']
            results[tech_id + ' Charge Option (kW)'] = \
                -self.variables_df['uch']
            results[tech_id + ' Discharge Option (kW)'] = \
                self.variables_df['udis']
            try:
                energy_rating = self.ene_max_rated.value
            except AttributeError:
                energy_rating = self.ene_max_rated

            results[tech_id + ' SOC (%)'] = \
                self.variables_df['ene'] / energy_rating

        return results

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
        return {f"{self.name.replace(' ', '_')}_dispatch_map": self.dispatch_map()}

    def dispatch_map(self):
        """ Takes the Net Power of the Storage System and tranforms it into a heat map

        Returns:

        """
        dispatch = pd.DataFrame(self.variables_df['dis'] - self.variables_df['ch'])
        dispatch.columns = ['Power']
        dispatch.loc[:, 'date'] = self.variables_df.index.date
        dispatch.loc[:, 'hour'] = (self.variables_df.index + pd.Timedelta('1s')).hour + 1
        dispatch = dispatch.reset_index(drop=True)
        dispatch_map = dispatch.pivot_table(values='Power', index='hour', columns='date')
        return dispatch_map

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.


        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        if self.variables_df.empty:
            return pro_forma
        analysis_years = self.variables_df.index.year.unique()
        tech_id = self.unique_tech_id()
        # OM COSTS
        om_costs = pd.DataFrame()
        dis = self.variables_df['dis']
        for year in analysis_years:
            # add fixed o&m costs
            index_yr = pd.Period(year=year, freq='y')
            # add fixed o&m costs
            try:
                fixed_om = -self.get_fixed_om().value
            except AttributeError:
                fixed_om = -self.get_fixed_om()
            om_costs.loc[index_yr, self.fixed_column_name()] = fixed_om
            # add variable o&m costs
            dis_sub = dis.loc[dis.index.year == year]
            om_costs.loc[index_yr, tech_id + ' Variable O&M Cost'] = -self.variable_om * self.dt * np.sum(dis_sub)

            # add startup costs
            if self.incl_startup:
                start_c_sub = self.variables_df['start_c'].loc[self.variables_df['start_c'].index.year == year]
                om_costs.loc[index_yr, tech_id + ' Start Charging Costs'] = -np.sum(start_c_sub * self.p_start_ch)
                start_d_sub = self.variables_df['start_d'].loc[self.variables_df['start_d'].index.year == year]
                om_costs.loc[index_yr, tech_id + ' Start Discharging Costs'] = -np.sum(start_d_sub * self.p_start_dis)

        # fill forward
        om_costs = fill_forward_func(om_costs, None)
        # apply inflation rates
        om_costs = apply_inflation_rate_func(om_costs, None, min(analysis_years))
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        return pro_forma

    def verbose_results(self):
        """ Results to be collected iff verbose -- added to the opt_results df

        Returns: a DataFrame

        """
        results = pd.DataFrame(index=self.variables_df.index)
        return results
