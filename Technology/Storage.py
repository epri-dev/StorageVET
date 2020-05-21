"""
Storage

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""

__author__ = 'Miles Evans, Evan Giarta and Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import copy
import logging
import cvxpy as cvx
import numpy as np
import pandas as pd
import re
import sys
from .DER import DER
try:
    import Constraint as Const
    import Library as Lib
except ModuleNotFoundError:
    import storagevet.Constraint as Const
    import storagevet.Library as Lib

u_logger = logging.getLogger('User')
e_logger = logging.getLogger('Error')


class Storage(DER):
    """ A general template for storage object

    We define "storage" as anything that can affect the quantity of load/power being delivered or used. Specific
    types of storage are subclasses. The storage subclass should be called. The storage class should never
    be called directly.

    """

    def __init__(self, name, params):
        """ Initialize all technology with the following attributes.

        Args:
            name (str): A unique string name for the technology being added
            params (dict): Dict of parameters
        """
        # create generic technology object
        DER.__init__(self, name, 'Energy Storage System', params)
        # input params
        # note: these should never be changed in simulation (i.e from degradation)
        self.rte = params['rte']/100
        self.sdr = params['sdr']
        self.ene_max_rated = params['ene_max_rated']
        self.dis_max_rated = params['dis_max_rated']
        self.dis_min_rated = params['dis_min_rated']
        self.ch_max_rated = params['ch_max_rated']
        self.ch_min_rated = params['ch_min_rated']
        self.ulsoc = params['ulsoc']/100
        self.llsoc = params['llsoc']/100
        self.soc_target = params['soc_target']/100
        self.ccost = params['ccost']
        self.ccost_kw = params['ccost_kw']
        self.ccost_kwh = params['ccost_kwh']
        self.fixedOM = params['fixedOM']  # DER has self.fixed_om
        self.OMexpenses = params['OMexpenses']  # DER has self.variable_om
        self.incl_startup = params['startup']
        self.incl_slack = params['slack']
        self.incl_binary = params['binary']
        self.daily_cycle_limit = params['daily_cycle_limit']

        if self.incl_startup:
            self.p_start_ch = params['p_start_ch']
            self.p_start_dis = params['p_start_dis']
        if self.incl_slack:
            self.kappa_ene_max = params['kappa_ene_max']
            self.kappa_ene_min = params['kappa_ene_min']
            self.kappa_ch_max = params['kappa_ch_max']
            self.kappa_ch_min = params['kappa_ch_min']
            self.kappa_dis_max = params['kappa_dis_max']
            self.kappa_dis_min = params['kappa_dis_min']

        # create physical constraints from input parameters TODO: remove 'rating' from names --HN
        # note: these can be changed throughout simulation (i.e. from degradation)
        self.physical_constraints = {'ene_min_rated': Const.Constraint('ene_min_rated', self.name, self.llsoc*self.ene_max_rated),
                                     'ene_max_rated': Const.Constraint('ene_max_rated', self.name, self.ulsoc*self.ene_max_rated),
                                     'ch_min_rated': Const.Constraint('ch_min_rated', self.name, self.ch_min_rated),
                                     'ch_max_rated': Const.Constraint('ch_max_rated', self.name, self.ch_max_rated),
                                     'dis_min_rated': Const.Constraint('dis_min_rated', self.name, self.dis_min_rated),
                                     'dis_max_rated': Const.Constraint('dis_max_rated', self.name, self.dis_max_rated)}

        self.fixed_om = self.fixedOM*self.dis_max_rated
        self.capex = self.ccost + (self.ccost_kw * self.dis_max_rated) + (self.ccost_kwh * self.ene_max_rated)
        # self.variable_names = {'ene', 'dis', 'ch'}
        self.variable_names = {'ene', 'dis', 'ch', 'ene_max_slack', 'ene_min_slack', 'dis_max_slack',
                               'dis_min_slack', 'ch_max_slack', 'ch_min_slack'}

    def calculate_control_constraints(self, datetimes):
        """ Generates a list of master or 'control constraints' from all predispatch service constraints.

        Args:
            datetimes (list): The values of the datetime column within the initial time_series data frame.

        Returns:
            Returns the physical constraints of the battery in timeseries form within a DataFrame.

        Note: the returned failed array returns the first infeasibility found, not all feasibilities.
        TODO: come back and check the user inputted constraints --HN
        """
        # create temp dataframe with values from physical_constraints
        temp_constraints = pd.DataFrame(index=datetimes)
        # create a df with all physical constraint values
        for constraint in self.physical_constraints.values():
            temp_constraints[re.search('^.+_.+_', constraint.name).group(0)[0:-1]] = copy.deepcopy(constraint.value)

        # change physical constraint with predispatch service constraints at each timestep
        # predispatch service constraints should be absolute constraints
        for service in self.predispatch_services.values():
            for constraint in service.constraints.values():
                if constraint.value is not None:
                    strp = constraint.name.split('_')
                    const_name = strp[0]
                    const_type = strp[1]
                    name = const_name + '_' + const_type
                    absolute_const = constraint.value.values  # constraint values
                    absolute_index = constraint.value.index  # the datetimes for which the constraint applies

                    current_const = temp_constraints.loc[absolute_index, name].values  # value of the current constraint

                    if const_type == "min":
                        # if minimum constraint, choose higher constraint value
                        temp_constraints.loc[absolute_index, name] = np.maximum(absolute_const, current_const)
                        # temp_constraints.loc[constraint.value.index, name] += constraint.value.values

                        # if the minimum value needed is greater than the physical maximum, infeasible scenario
                        max_value = self.physical_constraints[const_name + '_max' + '_rated'].value
                        if any(temp_constraints[name] > max_value):
                            return temp_constraints[temp_constraints[name] > max_value].index

                    else:
                        # if maximum constraint, choose lower constraint value
                        min_value = self.physical_constraints[const_name + '_min' + '_rated'].value
                        temp_constraints.loc[absolute_index, name] = np.minimum(absolute_const, current_const)
                        # temp_constraints.loc[constraint.value.index, name] -= constraint.value.values

                        if (const_name == 'ene') & any(temp_constraints[name] < min_value):
                            # if the maximum energy needed is less than the physical minimum, infeasible scenario
                            return temp_constraints[temp_constraints[name] > min_value].index
                        else:
                            # it is ok to floor at zero since negative power max values will be handled in power min
                            # i.e negative ch_max means dis_min should be positive and ch_max should be 0)
                            temp_constraints[name] = temp_constraints[name].clip(lower=0)

        # now that we have a new list of constraints, create Constraint objects and store as 'control constraint'
        self.control_constraints = {'ene_min': Const.Constraint('ene_min', self.name, temp_constraints['ene_min']),
                                    'ene_max': Const.Constraint('ene_max', self.name, temp_constraints['ene_max']),
                                    'ch_min': Const.Constraint('ch_min', self.name, temp_constraints['ch_min']),
                                    'ch_max': Const.Constraint('ch_max', self.name, temp_constraints['ch_max']),
                                    'dis_min': Const.Constraint('dis_min', self.name, temp_constraints['dis_min']),
                                    'dis_max': Const.Constraint('dis_max', self.name, temp_constraints['dis_max'])}

    def add_vars(self, size):
        """ Adds optimization variables to dictionary

        Variables added:
            ene (Variable): A cvxpy variable for Energy at the end of the time step
            dis (Variable): A cvxpy variable for Discharge Power, kW during the previous time step
            ch (Variable): A cvxpy variable for Charge Power, kW during the previous time step
            ene_max_slack (Variable): A cvxpy variable for energy max slack
            ene_min_slack (Variable): A cvxpy variable for energy min slack
            ch_max_slack (Variable): A cvxpy variable for charging max slack
            ch_min_slack (Variable): A cvxpy variable for charging min slack
            dis_max_slack (Variable): A cvxpy variable for discharging max slack
            dis_min_slack (Variable): A cvxpy variable for discharging min slack

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """

        variables = {'ene': cvx.Variable(shape=size, name='ene'),
                     'dis': cvx.Variable(shape=size, name='dis'),
                     'ch': cvx.Variable(shape=size, name='ch'),
                     'ene_max_slack': cvx.Parameter(shape=size, name='ene_max_slack', value=np.zeros(size)),
                     'ene_min_slack': cvx.Parameter(shape=size, name='ene_min_slack', value=np.zeros(size)),
                     'dis_max_slack': cvx.Parameter(shape=size, name='dis_max_slack', value=np.zeros(size)),
                     'dis_min_slack': cvx.Parameter(shape=size, name='dis_min_slack', value=np.zeros(size)),
                     'ch_max_slack': cvx.Parameter(shape=size, name='ch_max_slack', value=np.zeros(size)),
                     'ch_min_slack': cvx.Parameter(shape=size, name='ch_min_slack', value=np.zeros(size)),
                     'on_c': cvx.Parameter(shape=size, name='on_c', value=np.ones(size)),
                     'on_d': cvx.Parameter(shape=size, name='on_d', value=np.ones(size)),
                     }

        if self.incl_slack:
            self.variable_names.update(['ene_max_slack', 'ene_min_slack', 'dis_max_slack', 'dis_min_slack', 'ch_max_slack', 'ch_min_slack'])
            variables.update({'ene_max_slack': cvx.Variable(shape=size, name='ene_max_slack'),
                              'ene_min_slack': cvx.Variable(shape=size, name='ene_min_slack'),
                              'dis_max_slack': cvx.Variable(shape=size, name='dis_max_slack'),
                              'dis_min_slack': cvx.Variable(shape=size, name='dis_min_slack'),
                              'ch_max_slack': cvx.Variable(shape=size, name='ch_max_slack'),
                              'ch_min_slack': cvx.Variable(shape=size, name='ch_min_slack')})
        if self.incl_binary:
            self.variable_names.update(['on_c', 'on_d'])
            variables.update({'on_c': cvx.Variable(shape=size, boolean=True, name='on_c'),
                              'on_d': cvx.Variable(shape=size, boolean=True, name='on_d')})
            if self.incl_startup:
                self.variable_names.update(['start_c', 'start_d'])
                variables.update({'start_c': cvx.Variable(shape=size, name='start_c'),
                                  'start_d': cvx.Variable(shape=size, name='start_d')})

        return variables

    def objective_function(self, variables, mask, annuity_scalar=1):
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            variables (Dict): dictionary of variables being optimized
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                    the entire project lifetime (only to be set iff sizing, else annuity_scalar should not affect the aobject function)

        Returns:
            self.costs (Dict): Dict of objective costs
        """

        # create objective expression for variable om based on discharge activity
        var_om = cvx.sum(variables['dis']) * self.OMexpenses * self.dt * 1e-3 * annuity_scalar

        self.costs = {
            'fixed_om': self.fixed_om * annuity_scalar,
            'var_om': var_om}

        # add slack objective costs. These try to keep the slack variables as close to 0 as possible
        if self.incl_slack:
            self.costs.update({
                'ene_max_slack': cvx.sum(self.kappa_ene_max * variables['ene_max_slack']),
                'ene_min_slack': cvx.sum(self.kappa_ene_min * variables['ene_min_slack']),
                'dis_max_slack': cvx.sum(self.kappa_dis_max * variables['dis_max_slack']),
                'dis_min_slack': cvx.sum(self.kappa_dis_min * variables['dis_min_slack']),
                'ch_max_slack': cvx.sum(self.kappa_ch_max * variables['ch_max_slack']),
                'ch_min_slack': cvx.sum(self.kappa_ch_min * variables['ch_min_slack'])})

        # add startup objective costs
        if self.incl_startup:
            self.costs.update({
                          'ch_startup': cvx.sum(variables['start_c']) * self.p_start_ch * annuity_scalar,
                          'dis_startup': cvx.sum(variables['start_d']) * self.p_start_dis * annuity_scalar})

        return self.costs

    def objective_constraints(self, variables, mask, reservations, mpc_ene=None):
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            variables (Dict): Dictionary of variables being optimized
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set
            reservations (Dict): Dictionary of energy and power reservations required by the services being
                preformed with the current optimization subset
            mpc_ene (float): value of energy at end of last opt step (for mpc opt)

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """

        constraint_list = []

        size = int(np.sum(mask))

        curr_e_cap = self.physical_constraints['ene_max_rated'].value
        ene_target = self.soc_target * curr_e_cap

        # optimization variables
        ene = variables['ene']
        dis = variables['dis']
        ch = variables['ch']
        on_c = variables['on_c']
        on_d = variables['on_d']
        try:
            pv_gen = variables['pv_out']
        except KeyError:
            pv_gen = np.zeros(size)
        try:
            ice_gen = variables['ice_gen']
        except KeyError:
            ice_gen = np.zeros(size)

        # create cvx parameters of control constraints (this improves readability in cvx costs and better handling)
        ene_max = cvx.Parameter(size, value=self.control_constraints['ene_max'].value[mask].values, name='ene_max')
        ene_min = cvx.Parameter(size, value=self.control_constraints['ene_min'].value[mask].values, name='ene_min')
        ch_max = cvx.Parameter(size, value=self.control_constraints['ch_max'].value[mask].values, name='ch_max')
        ch_min = cvx.Parameter(size, value=self.control_constraints['ch_min'].value[mask].values, name='ch_min')
        dis_max = cvx.Parameter(size, value=self.control_constraints['dis_max'].value[mask].values, name='dis_max')
        dis_min = cvx.Parameter(size, value=self.control_constraints['dis_min'].value[mask].values, name='dis_min')

        # energy at the end of the last time step (makes sure that the end of the last time step is ENE_TARGET
        # TODO: rewrite this if MPC_ENE is not None
        constraint_list += [cvx.Zero((ene_target - ene[-1]) - (self.dt * ch[-1] * self.rte) + (self.dt * dis[-1]) - reservations['E'][-1] + (self.dt * ene[-1] * self.sdr * 0.01))]

        # energy generally for every time step
        constraint_list += [cvx.Zero(ene[1:] - ene[:-1] - (self.dt * ch[:-1] * self.rte) + (self.dt * dis[:-1]) - reservations['E'][:-1] + (self.dt * ene[:-1] * self.sdr * 0.01))]

        # energy at the beginning of the optimization window -- handles rolling window
        if mpc_ene is None:
            constraint_list += [cvx.Zero(ene[0] - ene_target)]
        else:
            constraint_list += [cvx.Zero(ene[0] - mpc_ene)]

        # Keep energy in bounds determined in the constraints configuration function -- making sure our storage meets control constraints
        constraint_list += [cvx.NonPos(ene_target - ene_max[-1] + reservations['E_upper'][-1] - variables['ene_max_slack'][-1])]
        constraint_list += [cvx.NonPos(ene[:-1] - ene_max[:-1] + reservations['E_upper'][:-1] - variables['ene_max_slack'][:-1])]

        constraint_list += [cvx.NonPos(-ene_target + ene_min[-1] + reservations['E_lower'][-1] - variables['ene_min_slack'][-1])]
        constraint_list += [cvx.NonPos(ene_min[1:] - ene[1:] + reservations['E_lower'][:-1] - variables['ene_min_slack'][:-1])]

        # Keep charge and discharge power levels within bounds
        constraint_list += [cvx.NonPos(-ch_max + ch - dis + reservations['D_min'] + reservations['C_max'] - variables['ch_max_slack'])]
        constraint_list += [cvx.NonPos(-ch + dis + reservations['C_min'] + reservations['D_max'] - dis_max - variables['dis_max_slack'])]

        constraint_list += [cvx.NonPos(ch - cvx.multiply(ch_max, on_c))]
        constraint_list += [cvx.NonPos(dis - cvx.multiply(dis_max, on_d))]

        # removing the band in between ch_min and dis_min that the battery will not operate in
        constraint_list += [cvx.NonPos(cvx.multiply(ch_min, on_c) - ch + reservations['C_min'])]
        constraint_list += [cvx.NonPos(cvx.multiply(dis_min, on_d) - dis + reservations['D_min'])]

        # the constraint below limits energy throughput and total discharge to less than or equal to
        # (number of cycles * energy capacity) per day, for technology warranty purposes
        # this constraint only applies when optimization window is equal to or greater than 24 hours
        if self.daily_cycle_limit and size >= 24:
            sub = mask.loc[mask]
            for day in sub.index.dayofyear.unique():
                day_mask = (day == sub.index.dayofyear)
                constraint_list += [cvx.NonPos(cvx.sum(dis[day_mask] * self.dt + cvx.pos(reservations['E'][day_mask]))
                                               - self.ene_max_rated * self.daily_cycle_limit)]
        elif self.daily_cycle_limit and size < 24:
            e_logger.info('Daily cycle limit did not apply as optimization window is less than 24 hours.')

        # constraints to keep slack variables positive
        if self.incl_slack:
            constraint_list += [cvx.NonPos(-variables['ch_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ch_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['dis_min_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_max_slack'])]
            constraint_list += [cvx.NonPos(-variables['ene_min_slack'])]

        if self.incl_binary:
            # when dis_min or ch_min has been overwritten (read: increased) by predispatch services, need to force technology to be on
            # TODO better way to do this???
            ind_d = [i for i in range(size) if self.control_constraints['dis_min'].value[mask].values[i] > self.physical_constraints['dis_min_rated'].value]
            ind_c = [i for i in range(size) if self.control_constraints['ch_min'].value[mask].values[i] > self.physical_constraints['ch_min_rated'].value]
            if len(ind_d) > 0:
                constraint_list += [on_d[ind_d] == 1]  # np.ones(len(ind_d))
            if len(ind_c) > 0:
                constraint_list += [on_c[ind_c] == 1]  # np.ones(len(ind_c))

            # note: cannot operate startup without binary
            if self.incl_startup:
                # startup variables are positive
                constraint_list += [cvx.NonPos(-variables['start_d'])]
                constraint_list += [cvx.NonPos(-variables['start_c'])]
                # difference between binary variables determine if started up in previous interval
                constraint_list += [cvx.NonPos(cvx.diff(on_d) - variables['start_d'][1:])]  # first variable not constrained
                constraint_list += [cvx.NonPos(cvx.diff(on_c) - variables['start_c'][1:])]  # first variable not constrained

        return constraint_list

    def calc_degradation(self, opt_period, start_dttm, end_dttm):
        """ Default is zero degradation. Only batteries can degrade over time.
        Args:
            start_dttm (DateTime): Start timestamp to calculate degradation
            end_dttm (DateTime): End timestamp to calculate degradation

        A percent that represented the energy capacity degradation
        """
        return 0

    def apply_degradation(self, datetimes):
        """ Default is no degradation effect. Only batteries can degrade over time.

        Args:
            datetimes (DateTime): Vector of timestamp to recalculate control_constraints

        Returns:
            Degraded energy capacity
        """
        pass

    def apply_past_degredation(self, mask, opt_period):
        """ Default is no degradation effect. Only batteries can degrade over time.

        Args:
            mask:
            opt_period:

        """
        pass

    def save_variable_results(self, variables, subs_index):
        """In addition to saving the variable results, this also makes sure that
        there is not charging and discharging at the same time

        Args:
            variables:
            subs_index:

        Returns:

        """
        super().save_variable_results(variables, subs_index)
        # GENERAL CHECKS ON SOLUTION
        # check for non zero slack
        if np.any(abs(self.variables.filter(regex="_*slack$")) >= 1):
            u_logger.warning('WARNING! non-zero slack variables found in optimization solution')
            e_logger.warning('WARNING! non-zero slack variables found in optimization solution')

    def timeseries_report(self):
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables.index)

        return results

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
        # recalculate fixed_om bc CBA might be evaluating a different fixedOM value or dis_max_rated might be a opt variable
        try:
            self.fixed_om = self.fixedOM * self.dis_max_rated.value
        except AttributeError:
            self.fixed_om = self.fixedOM * self.dis_max_rated

        pro_forma = super().proforma_report(opt_years, results)

        variable_col_name = self.name + ' Variable O&M Costs'
        dis = self.variables['dis']

        if self.incl_startup:
            start_c = self.variables['start_c']
            start_d = self.variables['start_d']

        for year in opt_years:
            # add variable o&m costs
            dis_sub = dis.loc[dis.index.year == year]
            pro_forma.loc[pd.Period(year=year, freq='y'), variable_col_name] = -self.OMexpenses * self.dt * np.sum(dis_sub) * 1e-3

            # add startup costs
            if self.incl_startup:
                start_c_sub = start_c.loc[start_c.index.year == year]
                pro_forma.loc[pd.Period(year=year, freq='y'), self.name + ' Start Charging Costs'] = -np.sum(start_c_sub * self.p_start_ch)
                start_d_sub = start_d.loc[start_d.index.year == year]
                pro_forma.loc[pd.Period(year=year, freq='y'), self.name + ' Start Discharging Costs'] = -np.sum(start_d_sub * self.p_start_dis)

        return pro_forma

    def verbose_results(self):
        """ Results to be collected iff verbose -- added to the opt_results df

        Returns: a DataFrame

        """
        results = pd.DataFrame(index=self.variables.index)
        for constraint in self.control_constraints.values():
            results.loc[:, 'Battery ' + constraint.name] = constraint.value

        return results
