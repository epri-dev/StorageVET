"""
Scenario.py

This Python class contains methods and attributes vital for completing the scenario analysis.
"""

__author__ = 'Halley Nathwani, Thien Nyguen, Miles Evans and Evan Giarta'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

from ValueStreams.DAEnergyTimeShift import DAEnergyTimeShift
from ValueStreams.FrequencyRegulation import FrequencyRegulation
from ValueStreams.NonspinningReserve import NonspinningReserve
from ValueStreams.DemandChargeReduction import DemandChargeReduction
from ValueStreams.EnergyTimeShift import EnergyTimeShift
from ValueStreams.SpinningReserve import SpinningReserve
from ValueStreams.Backup import Backup
from ValueStreams.Deferral import Deferral
from ValueStreams.DemandResponse import DemandResponse
from ValueStreams.ResourceAdequacy import ResourceAdequacy
from ValueStreams.UserConstraints import UserConstraints
from ValueStreams.VoltVar import VoltVar
from ValueStreams.LoadFollowing import LoadFollowing
from Technology.BatteryTech import BatteryTech
from Technology.CAESTech import CAESTech
from Technology.CurtailPV import CurtailPV
from Technology.ICE import ICE
from Technology.Load import Load
import numpy as np
import pandas as pd
import Finances as Fin
import cvxpy as cvx
import Library as Lib
from prettytable import PrettyTable
import time
import sys
import copy
import logging
from datetime import date
import calendar

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Scenario(object):
    """ A scenario is one simulation run in the model_parameters file.

    """

    def __init__(self, input_tree):
        """ Initialize a scenario.

        Args:
            input_tree (Params.Params): Params of input attributes such as time_series, params, and monthly_data

        """
        self.deferral_df = None  # Initialized to none -- if active, this will not be None!

        self.verbose = input_tree.Scenario['verbose']
        u_logger.info("Creating Scenario...")

        self.active_objects = {
            'value streams': [],
            'distributed energy resources': [],
        }

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
        self.incl_site_load = input_tree.Scenario['incl_site_load']
        self.incl_binary = input_tree.Scenario['binary']
        self.incl_slack = input_tree.Scenario['slack']
        # self.growth_rates = input_tree.Scenario['power_growth_rates']
        self.growth_rates = {'default': input_tree.Scenario['def_growth']}  # TODO: eventually turn into attribute
        self.frequency = input_tree.Scenario['frequency']

        self.no_export = input_tree.Scenario['no_export']
        self.no_import = input_tree.Scenario['no_import']

        self.customer_sided = input_tree.Scenario['customer_sided']

        self.technology_inputs_map = {
            'CAES': input_tree.CAES,
            'Battery': input_tree.Battery,
            'PV': input_tree.PV,
            'ICE': input_tree.ICE,
            'Load': input_tree.Load
        }

        self.predispatch_service_inputs_map = {
            'Deferral': input_tree.Deferral,
            'DR': input_tree.DR,
            'RA': input_tree.RA,
            'Backup': input_tree.Backup,
            'Volt': input_tree.Volt,
            'User': input_tree.User
        }
        self.service_input_map = {
            'DA': input_tree.DA,
            'FR': input_tree.FR,
            'LF': input_tree.LF,
            'SR': input_tree.SR,
            'NSR': input_tree.NSR,
            'DCM': input_tree.DCM,
            'retailTimeShift': input_tree.retailTimeShift,
        }

        self.solvers = set()

        # internal attributes to Case
        self.finance_inputs = input_tree.Finance
        self.services = {}
        self.predispatch_services = {}
        self.technologies = {}
        self.optimization_subsets = pd.DataFrame()
        self.objective_values = pd.DataFrame()

        u_logger.info("Scenario Created Successfully...")

    def add_technology(self):
        """ Reads params and adds technology. Each technology gets initialized and their physical constraints are found.

        """
        ess_action_map = {
            'Battery': BatteryTech,
            'CAES': CAESTech
        }

        for storage in ess_action_map.keys():  # this will cause merging errors -HN
            inputs = self.technology_inputs_map[storage]
            if inputs is not None:
                tech_func = ess_action_map[storage]
                self.technologies['Storage'] = tech_func('Storage', inputs)
                u_logger.info("Finished adding storage...")

        generator_action_map = {
            'PV': CurtailPV,
            'ICE': ICE
        }

        for gen in generator_action_map.keys():
            inputs = self.technology_inputs_map[gen]
            if inputs is not None:
                tech_func = generator_action_map[gen]
                new_gen = tech_func(gen, inputs)
                new_gen.estimate_year_data(self.opt_years, self.frequency)
                self.technologies[gen] = new_gen
        u_logger.info("Finished adding generators...")

        load_action_map = {
            'Load': Load
        }
        load_inputs = self.technology_inputs_map['Load']
        if load_inputs is not None:
            load_object = load_action_map['Load'](load_inputs)
            load_object.estimate_year_data(self.opt_years, self.frequency)
            self.technologies['Load'] = load_object
        self.active_objects['distributed energy resources'] = [self.technologies.keys()]
        u_logger.info("Finished adding active Technologies...")

    def add_services(self):
        """ Reads through params to determine which services are turned on or off. Then creates the corresponding
        service object and adds it to the list of services. Also generates a list of growth functions that apply to each
        service's timeseries data (to be used when adding growth data).

        Notes:
            This method needs to be applied after the technology has been initialized.
            ALL SERVICES ARE CONNECTED TO THE TECH

        """
        storage_inputs = self.technologies['Storage']

        predispatch_service_action_map = {
            'Deferral': Deferral,
            'DR': DemandResponse,
            'RA': ResourceAdequacy,
            'Backup': Backup,
            'Volt': VoltVar,
            'User': UserConstraints
        }
        for service, value in self.predispatch_service_inputs_map.items():
            if value is not None:
                u_logger.info("Using: " + str(service))
                inputs = self.predispatch_service_inputs_map[service]
                service_func = predispatch_service_action_map[service]
                new_service = service_func(inputs, self.technologies, self.dt)
                new_service.estimate_year_data(self.opt_years, self.frequency)
                self.predispatch_services[service] = new_service

        u_logger.info("Finished adding Predispatch Services for Value Stream")

        service_action_map = {
            'DA': DAEnergyTimeShift,
            'FR': FrequencyRegulation,
            'SR': SpinningReserve,
            'NSR': NonspinningReserve,
            'DCM': DemandChargeReduction,
            'retailTimeShift': EnergyTimeShift,
            'LF': LoadFollowing
        }

        for service, value in self.service_input_map.items():
            if value is not None:
                u_logger.info("Using: " + str(service))
                inputs = self.service_input_map[service]
                service_func = service_action_map[service]
                new_service = service_func(inputs, storage_inputs, self.dt)
                new_service.estimate_year_data(self.opt_years, self.frequency)
                self.services[service] = new_service

        self.active_objects['value streams'] = [*self.predispatch_services.keys()] + [*self.services.keys()]
        u_logger.info("Finished adding Services for Value Stream")

        # only execute check_for_deferral_failure() in self.add_services() here I.F.F Deferral is not the only option
        if 'Deferral' in self.active_objects['value streams']:
            self.opt_years = self.predispatch_services['Deferral'].check_for_deferral_failure(self.end_year, self.technologies, self.frequency, self.opt_years)

    def fill_and_drop_extra_data(self):
        """ Go through value streams and technologies and keep data for analysis years, and add more
        data if necessary.  ALSO creates/assigns optimization levels.

        Returns: None

        """
        # add missing years of data to each value stream
        for service in self.services.values():
            service.estimate_year_data(self.opt_years, self.frequency)

        for service in self.predispatch_services.values():
            service.estimate_year_data(self.opt_years, self.frequency)

        # remove any data that we will not use for analysis
        if 'PV' in self.technologies.keys():
            self.technologies['PV'].estimate_year_data(self.opt_years, self.frequency)
        if 'Load' in self.technologies.keys():
            self.technologies['Load'].estimate_year_data(self.opt_years, self.frequency)

        # create optimization levels
        self.optimization_subsets = self.assign_optimization_level(self.opt_years, self.n, 0, self.frequency, self.dt)

        # initialize degredation module IF battery is included
        if 'Battery' in self.technologies.keys():
            self.technologies['Battery'].initialize_degredation(self.optimization_subsets)

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
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level
            elif control_horizon == 'month':
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = current_control_level + (sub.index.month - 1)
            else:
                # n is number of hours
                control_horizon = int(control_horizon)
                sub['ind'] = range(len(sub))
                # split year into groups of n days
                ind = (sub.ind // (control_horizon / dt)).astype(int) + 1
                # continue counting from previous year opt_agg
                level_df.loc[level_df.index.year == yr, 'control'] = ind + current_control_level
            current_control_level = max(level_df.control) + 1

        # predictive level can overlap multiple years & there can be 1+ per timestep
        if not predictive_horizon:
            # set to be the control horizon
            level_df['predictive'] = level_df.loc[:, 'control']
        else:
            # TODO this has not been tested yet -- HN sorry hmu and I will help
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

    def add_control_constraints(self, deferral_check=False):
        """ This function collects 'absolute' constraints from the active value streams. Absolute constraints are
        time series constraints that will override the energy storage’s physical constraints. Graphically, the
        constraints that result from this function will lay on top of the physical constraints of that ESS to determine
        the acceptable envelope of operation. Therefore resulting in tighter maximum and minimum power/energy constraints.

        We create one general exogenous constraint for charge min, charge max, discharge min, discharge max, energy min,
        and energy max.

        Args:
            deferral_check (bool): flag to return non feasible timestamps if running deferral feasbility analysis

        """
        tech = self.technologies['Storage']

        for service in self.predispatch_services.values():
            tech.add_value_streams(service, predispatch=True)  # just storage
        feasible_check = tech.calculate_control_constraints(self.optimization_subsets.index)  # should pass any user inputted constraints here

        if (feasible_check is not None) & (not deferral_check):
            # if not running deferral failure analysis and infeasible scenario then stop and tell user
            u_logger.error('Predispatch and Storage inputs results in infeasible scenario')
            e_logger.error('Predispatch and Storage inputs results in infeasible scenario while adding control constraints.')
            quit()
        elif deferral_check:
            # return failure dttm to deferral failure analysis
            u_logger.info('Returned feasible_check to deferral failure analysis.')
            e_logger.error('Returned feasible_check to deferral failure analysis.')
            return feasible_check
        else:
            u_logger.info("Control Constraints Successfully Created...")

    def optimize_problem_loop(self, alpha=1):
        """ This function selects on opt_agg of data in self.time_series and calls optimization_problem on it.

        Args:
            alpha (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        """
        # remove any data that we will not use for analysis
        for service in self.services.values():
            service.estimate_year_data(self.opt_years, self.frequency)
        for service in self.predispatch_services.values():
            service.estimate_year_data(self.opt_years, self.frequency)
        if 'PV' in self.technologies.keys():
            self.technologies['PV'].estimate_year_data(self.opt_years, self.frequency)

        if 'Deferral' in self.predispatch_services.keys() and not len(self.services.keys()):
            # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
            # then do not optimize (skip the optimization loop)
            u_logger.info("Only active Value Stream is Deferral, so not optimizations will run...")
            return
        u_logger.info("Preparing Optimization Problem...")

        # list of all optimization windows
        periods = pd.Series(copy.deepcopy(self.optimization_subsets.predictive.unique()))
        periods.sort_values()

        # # initialize optimization objective results dataframe
        # self.objective_values = pd.DataFrame(index=periods)
        storage = self.technologies['Storage']

        for opt_period in periods:

            # used to select rows from time_series relevant to this optimization window
            mask = self.optimization_subsets.predictive == opt_period

            # apply past degradation
            storage.apply_past_degredation(mask, opt_period)

            print(f"{time.strftime('%H:%M:%S')} Running Optimization Problem starting at {self.optimization_subsets.loc[mask].index[0]} hb") if self.verbose else None

            # run optimization and return optimal variable and objective costs
            variable_dic, objective_values = self.optimization_problem(mask, alpha)

            sub_index = self.optimization_subsets.loc[mask].index
            # collect optimal dispatch variables solutions
            # TODO no need to return variable_dic with POI implementation
            for value in self.services.values():
                value.save_variable_results(variable_dic, sub_index)
            for value in self.predispatch_services.values():
                value.save_variable_results(variable_dic, sub_index)
            for value in self.technologies.values():
                value.save_variable_results(variable_dic, sub_index)

            # collect actual energy contributions solutions from services
            for serv in self.services.values():
                if self.customer_sided:
                    temp_ene_df = pd.DataFrame({'ene': np.zeros(len(sub_index))}, index=sub_index)
                else:
                    sub_list = serv.e[-1].value.flatten('F')
                    temp_ene_df = pd.DataFrame({'ene': sub_list}, index=sub_index)
                serv.ene_results = pd.concat([serv.ene_results, temp_ene_df], sort=True)

            storage.calc_degradation(opt_period, sub_index[0], sub_index[-1])

            # update objective_values for current OPT_PERIOD to have an index of OPT_PERIOD
            objective_values.index = pd.Index([opt_period])
            # add objective expressions to financial obj_val
            self.objective_values = pd.concat([self.objective_values, objective_values])

    def optimization_problem(self, mask, annuity_scalar=1):
        """ Sets up and runs optimization on a subset of data. Called within a loop.

        Args:
            mask (DataFrame): DataFrame of booleans used, the same length as self.time_series. The value is true if the
                        corresponding column in self.time_series is included in the data to be optimized.
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            variable_values (DataFrame): Optimal dispatch variables for each timestep in optimization period.

        """

        opt_var_size = int(np.sum(mask))

        obj_const = []  # list of constraint costs (make this a dict for continuity??)
        variable_dic = {}  # Dict of optimization variables
        obj_expression = {}  # Dict of objective costs

        # default power and energy reservations (these could be filled in with optimization variables or costs below)
        power_reservations = np.array([0, 0, 0, 0])  # [c_max, c_min, d_max, d_min]
        energy_throughputs = [cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                              cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero'),
                              cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='zero')]  # [e_upper, e, e_lower]

        ##########################################################################
        # COLLECT OPTIMIZATION VARIABLES & POWER/ENERGY RESERVATIONS/THROUGHPUTS
        ########################################################################

        # add optimization variables for each technology
        # TODO [multi-tech] need to handle naming multiple optimization variables (i.e ch_1)
        for tech in self.technologies.values():
            variable_dic.update(tech.add_vars(opt_var_size))

        # calculate system generation
        generation = cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='gen_zero')
        if 'PV' in self.technologies.keys():
            generation += variable_dic['pv_out']
        if 'ICE' in self.technologies.keys():
            generation += variable_dic['ice_gen']
        if 'CHP' in self.technologies.keys():
            generation += variable_dic['chp_elec']
        # aggregate load
        load = cvx.Parameter(shape=opt_var_size, value=np.zeros(opt_var_size), name='load_zero')
        if 'Load' in self.technologies.keys():
            load = self.technologies['Load'].get_charge(mask)

        value_streams = {**self.services, **self.predispatch_services}

        for stream in value_streams.values():
            # add optimization variables associated with each service
            variable_dic.update(stream.add_vars(opt_var_size))

            temp_power, temp_energy = stream.power_ene_reservations(variable_dic, mask)
            # add power reservations associated with each service
            power_reservations = power_reservations + np.array(temp_power)
            # add energy throughput and reservation associated with each service
            energy_throughputs = energy_throughputs + np.array(temp_energy)

        reservations = {'C_max': power_reservations[0],
                        'C_min': power_reservations[1],
                        'D_max': power_reservations[2],
                        'D_min': power_reservations[3],
                        'E': energy_throughputs[1],  # this represents the energy throughput of a value stream
                        'E_upper': energy_throughputs[0],  # max energy reservation (or throughput if called upon)
                        'E_lower': energy_throughputs[2]}  # min energy reservation (or throughput if called upon)

        #################################################
        # COLLECT OPTIMIZATION CONSTRAINTS & OBJECTIVES #
        #################################################

        # ADD IMPORT AND EXPORT CONSTRAINTS
        if self.no_export:
            obj_const += [cvx.NonPos(variable_dic['dis'] - variable_dic['ch'] + generation - load)]
        if self.no_import:
            obj_const += [cvx.NonPos(-variable_dic['dis'] + variable_dic['ch'] - generation + load)]

        pf_reliability = 'Reliability' in value_streams.keys() and len(value_streams.keys()) == 1 and value_streams['Reliability'].post_facto_only
        # add any constraints added by value streams
        for stream in value_streams.values():
            # add objective expression associated with each service
            obj_expression.update(stream.objective_function(variable_dic, mask, load, generation, annuity_scalar))

            obj_const += stream.objective_constraints(variable_dic, mask, load, generation, reservations)

        # add any objective costs from tech and the main physical constraints
        for tech in self.technologies.values():
            if not pf_reliability:
                # add technology costs if not pf reliability only
                obj_expression.update(tech.objective_function(variable_dic, mask, annuity_scalar))

            obj_const += tech.objective_constraints(variable_dic, mask, reservations)

        # TODO: need to add optimization constraints associated with power reservations by technology(i.e. CHP) via POI

        obj = cvx.Minimize(sum(obj_expression.values()))
        prob = cvx.Problem(obj, obj_const)
        u_logger.info("Finished setting up the problem. Solving now.")

        try:  # TODO: better try catch statement --HN
            if prob.is_mixed_integer():
                # MBL: GLPK will solver to a default tolerance but can take a long time. Can use ECOS_BB which uses a branch and bound method
                # and input a tolerance but I have found that this is a more sub-optimal solution. Would like to test with Gurobi
                # information on solvers and parameters here: https://www.cvxpy.org/tgitstatutorial/advanced/index.html

                # prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB, mi_abs_eps=1, mi_rel_eps=1e-2, mi_max_iters=1000)
                start = time.time()
                prob.solve(verbose=self.verbose_opt, solver=cvx.GLPK_MI)
                end = time.time()
                u_logger.info("Time it takes for solver to finish: " + str(end - start))
            else:
                start = time.time()
                # ECOS is default sovler and seems to work fine here
                prob.solve(verbose=self.verbose_opt, solver=cvx.ECOS_BB)
                end = time.time()
                u_logger.info("ecos solver")
                u_logger.info("Time it takes for solver to finish: " + str(end - start))
        except cvx.error.SolverError as err:
            e_logger.error("Solver Error. Exiting...")
            u_logger.error("Solver Error. Exiting...")
            sys.exit(err)

        u_logger.info('Optimization problem was %s', str(prob.status))

        solution_found = True
        if prob.status == 'infeasible':
            # tell the user and throw an error specific to the problem being infeasible
            solution_found = False
            e_logger.error('Optimization problem was %s', str(prob.status))
            if self.verbose:
                print('Problem was INFEASIBLE. No solution found.')
            raise cvx.SolverError('Problem was infeasible. No solution found.')

        if prob.status == 'unbounded':
            solution_found = False
            # tell the user and throw an error specific to the problem being unbounded
            e_logger.error('Optimization problem was %s', str(prob.status))
            if self.verbose:
                print('Problem is UNBOUNDED. No solution found.')
            raise cvx.SolverError('Problem is unbounded. No solution found.')

        # save solver used
        self.solvers = self.solvers.union(prob.solver_stats.solver_name)

        cvx_types = (cvx.expressions.cvxtypes.expression(), cvx.expressions.cvxtypes.constant())
        # evaluate optimal objective expression
        obj_values = pd.DataFrame({name: [obj_expression[name].value if isinstance(obj_expression[name], cvx_types) else obj_expression[name]] for name in list(obj_expression)})

        return variable_dic, obj_values  # TODO no need to return variable_dic with POI implementation

    @staticmethod
    def search_schema_type(root, attribute_name):
        """ Looks in the schema XML for the type of the attribute. Used to print the instance summary for previsualization.

        Args:
            root (object): the root of the input tree
            attribute_name (str): attribute being searched for

        Returns: the type of the attribute, if found. otherwise it returns "other"

        """
        for child in root:
            attributes = child.attrib
            if attributes.get('name') == attribute_name:
                if attributes.get('type') is None:
                    return "other"
                else:
                    return attributes.get('type')

    def instance_summary(self, input_tree):
        """ Prints each specific instance of this class, if there is sensitivity analysis, in the user log.

        Args:
            input_tree (dict): the input tree from Params.py

        Notes:
            Not used, but meant for sensitivity analysis

        """
        tree = input_tree.xmlTree
        treeRoot = tree.getroot()
        schema = input_tree.schema_tree

        u_logger.info("Printing summary table for each scenario...")
        table = PrettyTable()
        table.field_names = ["Category", "Element", "Active?", "Property", "Analysis?",
                             "Value", "Value Type", "Sensitivity"]
        for element in treeRoot:
            schemaType = self.search_schema_type(schema.getroot(), element.tag)
            activeness = element.attrib.get('active')
            for property in element:
                table.add_row([schemaType, element.tag, activeness, property.tag, property.attrib.get('analysis'),
                        property.find('Value').text, property.find('Type').text, property.find('Sensitivity').text])

        u_logger.info("\n" + str(table))
        u_logger.info("Successfully printed summary table for class Scenario in log file")
