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

Third-Party Software
EPRI does not own any portion of the software that is attributed
below.

<CVXPY/1.1.11> - <Steven Diamond>, <diamond@cs.stanford.edu>
Copyright © 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CVXPY is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the
implied warranties of merchantability and fitness for a particular
purpose are disclaimed.

This software relies on CVXPY to interface with work(s) covered by the
following copyright and permission notice(s): 

GLPK 5.0 - Andrew Makhorin, mao@gnu.org
Copyright © 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
2010, 2011, 2012 Andrew Makhorin, Department for Applied Informatics,
Moscow Aviation Institute, Moscow, Russia. All rights reserved.

Licensed under GNU Public License v3.0; you may not use GLPK except in
compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/gpl-3.0.en.html.

GLPK is a free program and is provided by the copyright holders and
contributors "as is" and any express or implied warranties, including,
but not limited to, the implied warranties of merchantability and fitness
for a particular purpose are disclaimed.
"""
"""
Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import time
import copy
from datetime import date
import calendar
from storagevet.ValueStreams.DAEnergyTimeShift import DAEnergyTimeShift
from storagevet.ValueStreams.FrequencyRegulation import FrequencyRegulation
from storagevet.ValueStreams.NonspinningReserve import NonspinningReserve
from storagevet.ValueStreams.DemandChargeReduction import DemandChargeReduction
from storagevet.ValueStreams.EnergyTimeShift import EnergyTimeShift
from storagevet.ValueStreams.SpinningReserve import SpinningReserve
from storagevet.ValueStreams.Backup import Backup
from storagevet.ValueStreams.Deferral import Deferral
from storagevet.ValueStreams.DemandResponse import DemandResponse
from storagevet.ValueStreams.ResourceAdequacy import ResourceAdequacy
from storagevet.ValueStreams.UserConstraints import UserConstraints
from storagevet.ValueStreams.VoltVar import VoltVar
from storagevet.ValueStreams.LoadFollowing import LoadFollowing
from storagevet.Technology.BatteryTech import Battery
from storagevet.Technology.CAESTech import CAES
from storagevet.Technology.PVSystem import PV
from storagevet.Technology.InternalCombustionEngine import ICE
from storagevet.Technology.Load import Load
from storagevet.ServiceAggregator import ServiceAggregator
from storagevet.POI import POI
import storagevet.Finances as Fin
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class Scenario(object):
    """ A scenario is one simulation run in the model_parameters file.

    """
    TECH_CLASS_MAP = {
        'CAES': CAES,
        'Battery': Battery,
        'PV': PV,
        'ICE': ICE,
        'Load': Load
    }
    VS_CLASS_MAP = {
        'Deferral': Deferral,
        'DR': DemandResponse,
        'RA': ResourceAdequacy,
        'Backup': Backup,
        'Volt': VoltVar,  # THIS DOES NOT WORK
        'User': UserConstraints,
        'DA': DAEnergyTimeShift,
        'FR': FrequencyRegulation,
        'LF': LoadFollowing,
        'SR': SpinningReserve,
        'NSR': NonspinningReserve,
        'DCM': DemandChargeReduction,
        'retailTimeShift': EnergyTimeShift,
    }

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            input_tree (Params.Params): Params of input attributes such as time_series, params, and monthly_data

        """
        self.verbose = input_tree.Scenario['verbose']
        self.start_time = time.time()
        self.start_time_frmt = time.strftime('%Y%m%d%H%M%S')
        self.end_time = 0

        # add general case params (USER INPUTS)
        self.dt = input_tree.Scenario['dt']
        self.verbose_opt = input_tree.Scenario['verbose_opt']
        self.n = input_tree.Scenario['n']
        # self.n_control = input_tree.Scenario['n_control']
        self.n_control = 0
        self.mpc = input_tree.Scenario['mpc']

        self.start_year = input_tree.Scenario['start_year']
        self.end_year = input_tree.Scenario['end_year']
        self.opt_years = input_tree.Scenario['opt_years']
        self.incl_binary = input_tree.Scenario['binary']
        self.incl_slack = input_tree.Scenario['slack']
        self.def_growth = input_tree.Scenario['def_growth']/100
        self.frequency = input_tree.Scenario['frequency']

        # save inputs to be used to initialize classes later
        self.technology_inputs_map = {
            'CAES': input_tree.CAES,
            'Battery': input_tree.Battery,
            'PV': input_tree.PV,
            'ICE': input_tree.ICE,
            'Load': input_tree.Load
        }
        self.value_stream_input_map = {
            'Deferral': input_tree.Deferral,
            'DR': input_tree.DR,
            'RA': input_tree.RA,
            'Backup': input_tree.Backup,
            'Volt': input_tree.Volt,
            'User': input_tree.User,
            'DA': input_tree.DA,
            'FR': input_tree.FR,
            'LF': input_tree.LF,
            'SR': input_tree.SR,
            'NSR': input_tree.NSR,
            'DCM': input_tree.DCM,
            'retailTimeShift': input_tree.retailTimeShift,
        }
        self.poi_inputs = input_tree.POI
        self.finance_inputs = input_tree.Finance

        # these are attributes that are changed as the scenario is solved
        self.solvers = []
        self.poi = None
        self.service_agg = None
        self.cost_benefit_analysis = None
        self.optimization_levels = pd.DataFrame()
        self.objective_values = pd.DataFrame()
        self.system_requirements = None
        self.opt_engine = True  # indicates that dervet should go to the optimization module and size there

    def set_up_poi_and_service_aggregator(self, point_of_interconnection_class=POI, service_aggregator_class=ServiceAggregator):
        """ Initialize the POI and service aggregator with DERs and valuestreams to be evaluated.

        """
        # these need to be initialized after opt_agg is created
        self.poi = point_of_interconnection_class(self.poi_inputs, self.technology_inputs_map, self.TECH_CLASS_MAP)
        self.service_agg = service_aggregator_class(self.value_stream_input_map, self.VS_CLASS_MAP)
        if self.service_agg.is_deferral_only():
            TellUser.warning("Only active Value Stream is Deferral, so not optimizations will be solved...")
            self.opt_engine = False

    def initialize_cba(self):
        """ Initialize StorageVET's limited cost benefit analysis module with user given inputs
        """
        self.cost_benefit_analysis = Fin.Financial(self.finance_inputs, self.start_year, self.end_year)
        # add fuel_cost to active DERs that can consume fuel
        for der in self.poi.der_list:
            der.set_fuel_cost(self.cost_benefit_analysis.get_fuel_cost)

    def fill_and_drop_extra_data(self):
        """ Go through value streams and technologies and keep data for analysis years, and add more
        data if necessary.  ALSO creates/assigns optimization levels.

        Returns: None

        """
        self.opt_years = self.service_agg.update_analysis_years(self.end_year, self.poi, self.frequency, self.opt_years, self.def_growth)

        # add rte_list from all active ess to each value stream
        for service in self.service_agg.value_streams.values():
            service.rte_list(self.poi)

        # add missing years of data to each value stream
        for service in self.service_agg.value_streams.values():
            service.grow_drop_data(self.opt_years, self.frequency, self.def_growth)

        # remove any data that we will not use for analysis
        for der in self.poi.der_list:
            der.grow_drop_data(self.opt_years, self.frequency, self.def_growth)

        # create optimization levels
        self.optimization_levels = self.assign_optimization_level(self.opt_years, self.n, 0, self.frequency, self.dt)

        # initialize degredation module in ESS objects (NOTE: if no degredation module applies to specific ESS tech, then nothing happens)
        for der in self.poi.der_list:
            if der.technology_type == "Energy Storage System":
                der.initialize_degradation_module(self.optimization_levels)

        # calculate and check that system requirement set by value streams can be met
        self.system_requirements = self.service_agg.identify_system_requirements(self.poi.der_list, self.opt_years, self.frequency)

    @staticmethod
    def assign_optimization_level(analysis_years, control_horizon, predictive_horizon, frequency, dt):
        """ creates an index based on the opt_years presented and then

         Args:
            analysis_years (list): List of Period years where we need data for
            control_horizon (str, int): optimization window length from the user
            predictive_horizon (str, int): mcp horizon input from the user
                (if 0, then assume the same as CONTROL_HORIZON)
                should be greater than or equal to CONTROL_HORIZON value
            frequency (str): time step in string form
            dt (float): time step

        Return:
            opt_agg (DataFrame): 1 column, all indexes with the same value will be in one
            optimization problem together

        """
        # create dataframe to fill
        level_index = Lib.create_timeseries_index(analysis_years, frequency)
        level_df = pd.DataFrame({'control': np.zeros(len(level_index))}, index=level_index)
        current_control_level = 0
        # control level should not overlap multiple years & there is only one per timestep
        for yr in level_index.year.unique():
            sub = copy.deepcopy(level_df[level_df.index.year == yr])
            if control_horizon == 'year':
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level + 1
            elif control_horizon == 'month':
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level + sub.index.month
            else:
                # n is number of hours
                control_horizon = int(control_horizon)
                sub['ind'] = range(len(sub))
                # split year into groups of n days
                ind = (sub.ind // (control_horizon / dt)).astype(int) + 1
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = ind + current_control_level
            current_control_level = max(level_df.control)

        # predictive level can overlap multiple years & there can be 1+ per timestep
        if not predictive_horizon:
            # set to be the control horizon
            level_df['predictive'] = level_df.loc[:, 'control']
        else:
            # TODO this has not been tested yet -- HN (sorry hmu and I will help)
            # create a list of lists
            max_index = len(level_df['control'])
            predictive_level = np.repeat([], max_index)
            current_predictive_level_beginning = 0
            current_predictive_level = 0

            for control_level in level_df.control.unique():
                if predictive_horizon == 'year':
                    # needs to be a year from the beginning of the current predictive level, determine
                    # length of the year based on first index in subset
                    start_year = level_index[current_predictive_level_beginning].year[0]
                    f_date = date(start_year, 1, 1)
                    l_date = date(start_year + 1, 1, 1)
                    delta = l_date - f_date
                    current_predictive_level_end = int(delta.days*dt)

                elif predictive_horizon == 'month':
                    # needs to be a month from the beginning of the current predictive level, determine
                    # length of the month based on first index in subset
                    start_index = level_index[current_predictive_level_beginning]
                    current_predictive_level_end = calendar.monthrange(start_index.year, start_index.month)
                else:
                    current_predictive_level_end = predictive_horizon * dt
                # make sure that CURRENT_PREDICTIVE_LEVEL_END stays less than or equal to MAX_INDEX
                current_predictive_level_end = min(current_predictive_level_end, max_index)
                # add CURRENT_PREDICTIVE_LEVEL to lists between CURRENT_PREDICTIVE_LEVEL_BEGINNING and CURRENT_PREDICTIVE_LEVEL_END
                update_levels = predictive_level[current_predictive_level_beginning, current_predictive_level_end]
                update_levels = [dt_level.append(current_predictive_level) for dt_level in update_levels]
                predictive_level[current_predictive_level_beginning, current_predictive_level_end] = update_levels
                current_predictive_level_beginning = np.sum(level_df.control == control_level)
                # increase CURRENT_PREDICTIVE_LEVEL
                current_predictive_level += 1
            level_df['predictive'] = predictive_level
        return level_df

    def optimize_problem_loop(self):
        """ This function selects on opt_agg of data in time_series and calls optimization_problem on it.

        """
        if not self.opt_engine:
            return

        TellUser.info("Starting optimization loop")
        for opt_period in self.optimization_levels.predictive.unique():
            # setup + run optimization then return optimal objective costs
            functions, constraints, sub_index = self.set_up_optimization(opt_period)

            ##NOTE: these print statements reveal the final constraints and costs for debugging
            #print(f'\nFinal constraints ({len(constraints)}):')
            #print(f'\nconstraints ({len(constraints)}):')
            #print('\n'.join([f'{i}: {c}' for i, c in enumerate(constraints)]))
            #print(f'\ncosts ({len(functions)}):')
            #print('\n'.join([f'{k}: {v}' for k, v in functions.items()]))
            #print()

            cvx_problem, obj_expressions, cvx_error_msg = self.solve_optimization(functions, constraints)
            self.save_optimization_results(opt_period, sub_index, cvx_problem, obj_expressions, cvx_error_msg)

    def set_up_optimization(self, opt_window_num, annuity_scalar=1, ignore_der_costs=False):
        """ Sets up and runs optimization on a subset of time in a year. Called within a loop.

        Args:
            opt_window_num (int): the optimization window number that is being solved
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing OR optimizing carrying costs)
            ignore_der_costs (bool): flag to indicate if we do not want to consider to economics of operating the DERs in our optimization
                (this flag will never be TRUE if the user indicated the desire to size the DER mix)

        Returns:
            functions (dict): functions or objectives of the optimization
            constraints (list): constraints that define behaviors, constrain variables, etc. that the optimization must meet
            sub_index (pd.Index): index of the optimization window represented in our optimization

        """
        # used to select rows from time_series relevant to this optimization window
        mask = self.optimization_levels.predictive == opt_window_num
        sub_index = self.optimization_levels.loc[mask].index
        TellUser.info(f"{time.strftime('%H:%M:%S')} Running Optimization Problem starting at {sub_index[0]} hb")
        opt_var_size = int(np.sum(mask))

        # set up variables
        self.poi.initialize_optimization_variables(opt_var_size)
        self.service_agg.initialize_optimization_variables(opt_var_size)

        # grab values from the POI that is required to know calculate objective functions and constraints
        load_sum, var_gen_sum, gen_sum, tot_net_ess, der_dispatch_net_power, total_soe, agg_p_in, agg_p_out, agg_steam, agg_hotwater, agg_cold = self.poi.get_state_of_system(mask)
        combined_rating = self.poi.combined_discharge_rating_for_reliability()

        # set up controller first to collect and provide inputs to the POI
        funcs, consts = self.service_agg.optimization_problem(mask, load_sum, var_gen_sum, gen_sum, tot_net_ess, combined_rating, annuity_scalar)

        # add optimization problem portion from the POI
        temp_objectives, temp_consts = self.poi.optimization_problem(mask, agg_p_in, agg_p_out, agg_steam, agg_hotwater, agg_cold, annuity_scalar)
        if not ignore_der_costs:
            #  don't ignore der costs
            funcs.update(temp_objectives)
        consts += temp_consts

        # add system requirement constraints (get the subset of data that applies to the current optimization window)
        for req_name, requirement in self.system_requirements.items():

            # NOTE: der_dispatch_net_power is (charge - discharge) for each DER that can dispatch power
            #           (not Intermittent Resources, and not Load)
            if req_name == 'der dispatch discharge min':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='DerDispatchDisMinReq')
                consts += [cvx.NonPos(req_parameter + der_dispatch_net_power)]
                continue

            #if req_name == 'der dispatch charge max':
            #    req_value = requirement.get_subset(mask)
            #    #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
            #    req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='DerDispatchChMaxReq')
            #    consts += [cvx.NonPos(der_dispatch_net_power + -1 * req_parameter)]
            #    continue

            if req_name == 'poi export min':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiExportMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * agg_p_out)]
                continue

            if req_name == 'poi export max':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiExportMaxReq')
                consts += [cvx.NonPos(agg_p_out + -1 * req_parameter)]
                continue

            if req_name == 'poi import min':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiImportMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * agg_p_in)]
                continue

            if req_name == 'poi import max':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='PoiImportMaxReq')
                consts += [cvx.NonPos(agg_p_in + -1 * req_parameter)]
                continue

            if req_name == 'energy min':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='SysEneMinReq')
                consts += [cvx.NonPos(req_parameter + -1 * total_soe)]
                continue

            if req_name == 'energy max':
                req_value = requirement.get_subset(mask)
                #print(f'{req_name} (range):\n{req_value.min()} -- {req_value.max()}')
                req_parameter = cvx.Parameter(shape=opt_var_size, value=req_value, name='SysEneMaxReq')
                consts += [cvx.NonPos(total_soe + -1 * req_parameter)]
                continue

            # if this part of the method is reached, we have failed to recognize a system requirement and should fail
            error_message = f'This system requirement: "{req_name}" is not properly specified by the Scenario class. Cannot continue.'
            TellUser.error(error_message)
            raise SystemRequirementsError(error_message)

        res_dis_d, res_dis_u, res_ch_d, res_ch_u, ue_prov, ue_stor, worst_ue_pro, worst_ue_sto = self.service_agg.aggregate_reservations(mask)
        sch_dis_d, sch_dis_u, sch_ch_d, sch_ch_u, ue_decr, ue_incr, total_dusoe = self.poi.aggregate_p_schedules(mask)
        # make sure P schedule matches the P reservations
        consts += [cvx.NonPos(res_dis_d + (-1) * sch_dis_d)]
        consts += [cvx.NonPos(res_dis_u + (-1) * sch_dis_u)]
        consts += [cvx.NonPos(res_ch_u + (-1) * sch_ch_u)]
        consts += [cvx.NonPos(res_ch_d + (-1) * sch_ch_d)]

        # match uE delta to uE reservation: energy increase
        consts += [cvx.Zero(ue_prov + (-1) * ue_decr)]
        # match uE delta to uE reservation: energy decrease
        consts += [cvx.Zero(ue_stor + (-1) * ue_incr)]
        # make sure that the net change in energy is less than the total change in system SOE
        consts += [cvx.NonPos(total_dusoe + (-1) * ue_prov + (-1) * ue_stor)]

        # require that SOE +/- worst case stays within bounds of DER mix
        _, _, soe_limits = self.poi.calculate_system_size()
        consts += [cvx.NonPos(total_soe + worst_ue_sto - soe_limits[0])]
        consts += [cvx.NonPos(soe_limits[1] + worst_ue_pro + (-1)*total_soe)]

        return funcs, consts, sub_index

    def solve_optimization(self, obj_expression, obj_const, force_glpk_mi=False):
        """ Sets up and runs optimization on a subset of time in a year. Called within a loop.

        Args:
            obj_expression (dict): functions or objectives of the optimization
            obj_const (list): constraints that define behaviors, constrain variables, etc. that the optimization must meet

        Returns:
            objective_values (DataFrame): the minimum values for each cost function subject to constraints

        """
        # summary of objective expressions to set up optimization problem
        obj = cvx.Minimize(sum(obj_expression.values()))
        prob = cvx.Problem(obj, obj_const)
        TellUser.info("Finished setting up the problem. Solving now.")
        cvx_error_msg = ''
        try:
            if prob.is_mixed_integer():
                # MBL: GLPK will solver to a default tolerance but can take a long time. Can use ECOS_BB which uses a branch and bound method
                # and input a tolerance but I have found that this is a more sub-optimal solution. Would like to test with Gurobi
                # information on solvers and parameters here: https://www.cvxpy.org/tgitstatutorial/advanced/index.html

                # prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB, mi_abs_eps=1, mi_rel_eps=1e-2, mi_max_iters=1000)
                start = time.time()
                TellUser.debug("glpk_mi solver")
                prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                end = time.time()
                TellUser.info("Time it takes for solver to finish: " + str(end - start))
            else:
                start = time.time()
                # ECOS is default solver and seems to work fine here, however...
                # a problem with ECOS was found when running projects with thermal loads,
                # so we force use of glpk_mi for these cases
                if force_glpk_mi:
                    TellUser.debug("glpk_mi solver (for cases with thermal loads)")
                    prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                else:
                    TellUser.debug("ecos_bb solver")
                    prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB)
                end = time.time()
                TellUser.info("Time it takes for solver to finish: " + str(end - start))
        except (cvx.error.SolverError, RuntimeError) as e:
            TellUser.error("The solver was unable to find a solution.")
            cvx_error_msg = e
        return prob, obj_expression, cvx_error_msg

    def save_optimization_results(self, opt_window_num, sub_index, prob, obj_expression, cvx_error_msg):
        """ Checks if there was a solution to the optimization. If not, report the problem
         to the user. If there was a solution, then saves results within each instance.

        Args:
            opt_window_num:
            sub_index:
            prob:
            obj_expression:
            cvx_error_msg: any error message that might have occurred during problem solve

        """
        TellUser.info(f'Optimization problem was {prob.status}')
        # save solver used
        try:
            self.solvers.append(prob.solver_stats.solver_name)
        except AttributeError:
            pass

        if (prob.status == 'infeasible') or (prob.status == 'unbounded') or (prob.status is None):
            # tell the user and throw an error specific to the problem being infeasible/unbounded
            error_msg = f'Optimization window {opt_window_num} was {prob.status}. No solution found. Look in *.log for for information'
            TellUser.error(cvx_error_msg)
            if prob.status == 'infeasible':
                raise SolverInfeasibleError(error_msg)
            elif prob.status == 'unbounded':
                raise SolverUnboundedError(error_msg)
            else:
                raise SolverError(error_msg)
        # evaluate optimal objective expression
        for cost, func in obj_expression.items():
            try:
                obj_expression[cost] = func.value
            except AttributeError:
                continue

        obj_values = pd.DataFrame(obj_expression, index=[opt_window_num])
        # then add objective expressions to financial obj_val
        self.objective_values = pd.concat([self.objective_values, obj_values])

        # GENERAL CHECK ON SOLUTION: check for non zero slack
        if np.any(abs(obj_values.filter(regex="_*slack$")) >= 1):
            TellUser.warning('non-zero slack variables found in optimization solution')
        for vs in self.service_agg.value_streams.values():
            vs.save_variable_results(sub_index)

        for der in self.poi.active_ders:
            # record the solution of the variables and run again
            der.save_variable_results(sub_index)
            # calculate degradation in Battery instances
            if der.tag == "Battery":
                der.calc_degradation(opt_window_num, sub_index[0], sub_index[-1])
