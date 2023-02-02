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
import pandas as pd
import numpy as np
import cvxpy as cvx
from storagevet.SystemRequirement import SystemRequirement
from storagevet.ErrorHandling import *


class ServiceAggregator:
    """ Tracks the value streams and bids the storage's capabilities into markets
    """

    def __init__(self, value_stream_inputs_map, value_stream_class_map):
        """
        Create the corresponding service object and add it to the list of services.
        Also generates a list of growth functions that apply to each service's timeseries data (to be used when adding growth data).

        Args:
            value_stream_inputs_map (Dict): dictionary of all active value streams in this format:
                'value_stream_name': Value_Stream_object

        """

        self.value_streams = {}

        for service, params_input in value_stream_inputs_map.items():
            if params_input is not None:  # then Params class found an input
                TellUser.info("Initializing: " + str(service))
                self.value_streams[service] = value_stream_class_map[service](params_input)
        TellUser.debug("Finished adding value streams")

        self.sys_requirements = {}
        self.system_requirements_conflict = False

    def update_analysis_years(self, end_year, poi, frequency, opt_years, def_load_growth):
        if 'Deferral' in self.value_streams.keys():
            return self.value_streams['Deferral'].check_for_deferral_failure(end_year, poi, frequency, opt_years, def_load_growth)
        return opt_years

    def is_deferral_only(self):
        """ if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
        then do not optimize (skip the optimization loop)

        Returns: Bool that will skip the optimization loop and any cakcukations that depend on an
        optimization solution

        """
        return 'Deferral' in self.value_streams and len(self.value_streams) == 1

    def initialize_optimization_variables(self, size):
        """Function should be called at the beginning of each optimization loop. Initializes optimization variables

        Args:
            size (int): length of optimization variables_df to create
        """
        for value_stream in self.value_streams.values():
            value_stream.initialize_variables(size)

    def identify_system_requirements(self, der_lst, years_of_analysis, datetime_freq):
        """ This function collects system requirements from the active value streams. This function only
        needs to be called once if than index for each optimization year that the optimization needs to be
        run for

        Args:
            der_lst (list): list of the initialized DERs in our scenario
            years_of_analysis (list): list of years that should be included in the returned Index
            datetime_freq (str): the pandas frequency in string representation -- required to create dateTime range

        Returns: dict of min/max ch-dis-soe requirements set by system requirements
        """
        for service in self.value_streams.values():
            service.calculate_system_requirements(der_lst)
            for constraint in service.system_requirements:
                # check to see if system requirement has been initialized
                limit_key = f"{constraint.type} {constraint.limit_type}"
                sys_requ = self.sys_requirements.get(limit_key)
                if sys_requ is None:
                    # if not, then initialize one
                    sys_requ = SystemRequirement(constraint.type, constraint.limit_type, years_of_analysis, datetime_freq)
                # update system requirement
                sys_requ.update(constraint)
                # save system requirement
                self.sys_requirements[limit_key] = sys_requ

        # report the datetimes and VS that contributed to the conflict
        # 1) check if poi import max and poi import min conflict
        if self.sys_requirements.get('poi import min') is not None and self.sys_requirements.get('poi import max') is not None:
            poi_import_conflicts = self.sys_requirements.get('poi import min') > self.sys_requirements.get('poi import max')
            self.report_conflict(poi_import_conflicts, ['poi import min', 'poi import max'])
        # 2) check if energy max and energy min conflict
        if self.sys_requirements.get('energy min') is not None and self.sys_requirements.get('energy max') is not None:
            energy_conflicts = self.sys_requirements.get('energy min') > self.sys_requirements.get('energy max')
            self.report_conflict(energy_conflicts, ['energy min', 'energy max'])
        # 3) check if poi export max and poi export min conflict
        if self.sys_requirements.get('poi export min') is not None and self.sys_requirements.get('poi export max') is not None:
            poi_export_conflicts = self.sys_requirements.get('poi export min') > self.sys_requirements.get('poi export max')
            self.report_conflict(poi_export_conflicts, ['poi export min', 'poi export max'])
        # 4) check if poi export min and poi import min conflict (cannot be > 0 (nonzero at input) at the same time)
        if self.sys_requirements.get('poi import min') is not None and self.sys_requirements.get('poi export min') is not None:
            poi_import_and_poi_export_conflicts = (self.sys_requirements.get('poi import min') > 0) & (self.sys_requirements.get('poi export min') > 0)
            self.report_conflict(poi_import_and_poi_export_conflicts, ['poi import min', 'poi export min'])
        if self.system_requirements_conflict:
            raise SystemRequirementsError('System requirements are not possible. Check log files for more information.')
        else:
            return self.sys_requirements

    def report_conflict(self, conflict_mask, check_sys_req):
        """ Checks to see if there was a conflict, if so then report it to the user and flag that the optimization
        will not be able to solve (so to stop running)

        Args:
            conflict_mask (DataFrame): A boolean array that is true for indices corresponding to conflicts that occurs
            check_sys_req (list): the sys reqs to check

        """
        if np.any(conflict_mask):
            self.system_requirements_conflict = True
            datetimes = conflict_mask.index[conflict_mask]
            if len(datetimes):
                TellUser.error(f'System requirements are not possible at {datetimes.to_list()}')
                for req in check_sys_req:
                    TellUser.error(f"The following contribute to the {req} error: {self.sys_requirements.get(req).contributors(datetimes)}")

    def optimization_problem(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating, annuity_scalar=1):
        """ Generates the full objective function, including the optimization variables.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (Expression): the sum of load within the system
            generator_out_sum (Expression): the sum of conventional generation within the system
            net_ess_power (Expression): the sum of the net power of all the ESS in the system. [= charge - discharge]
            combined_rating (cvx.Expression, int): the combined rating of DER that can reliabily dispatched in a worst case situtation
                these usually tend to be ESS and Generators
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            A dictionary with the portion of the objective function that it affects, labeled by the expression's key.
            A list of optimization constraints.
        """
        opt_functions = {}
        opt_constraints = []
        for value_stream in self.value_streams.values():
            opt_functions.update(value_stream.objective_function(mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, annuity_scalar))
            opt_constraints += value_stream.constraints(mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power, combined_rating)
        return opt_functions, opt_constraints

    def aggregate_reservations(self, mask):
        """Calculates:
            - total amount of power (charging) that needs to be reserved for this value stream
            - total amount of power (discharging) that needs to be reserved for this value stream
            - total uE throughput
            - worst case uE throughput

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            reservation to pull power down from the grid by discharging less
            reservation to push power up into the grid by discharging more
            reservation to pull power down from the grid by charging more
            reservation to push power up into the grid by charging less
            energy reservation stored/delivered bc of sub-time-step activities
            worst case energy provided due to sub-time-step activities
            worst case energy stored due to sub-time-step activities
        """
        charge_up = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        charge_down = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        discharge_up = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        discharge_down = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        uenergy_stored = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        uenergy_provided = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        worst_ue_stored = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')
        worst_ue_provided = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')

        for value_stream in self.value_streams.values():
            charge_up += value_stream.p_reservation_charge_up(mask)
            charge_down += value_stream.p_reservation_charge_down(mask)
            discharge_up += value_stream.p_reservation_discharge_up(mask)
            discharge_down += value_stream.p_reservation_discharge_down(mask)
            uenergy_stored += value_stream.uenergy_option_stored(mask)
            uenergy_provided += value_stream.uenergy_option_provided(mask)
            worst_ue_stored += value_stream.worst_case_uenergy_stored(mask)
            worst_ue_provided += value_stream.worst_case_uenergy_provided(mask)

        return discharge_down, discharge_up, charge_down, charge_up, uenergy_provided, uenergy_stored, worst_ue_provided, worst_ue_stored

    def save_optimization_results(self, subs_index):
        """This function should be called after each optimization problem is solved. Saves the values at which the optimization functions
        evaluate to the minimum.

        Args:
            subs_index (Index): index of the subset of data for which the variables_df were solved for

        """
        for value_stream in self.value_streams.values():
            value_stream.save_variable_results(subs_index)

    def merge_reports(self):
        """ Collects and merges the optimization results for all Value Streams into a DataFrame

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        results = pd.DataFrame()
        monthly_data = pd.DataFrame()

        for service in self.value_streams.values():
            report_df = service.timeseries_report()
            results = pd.concat([results, report_df], axis=1, sort=False)
            report = service.monthly_report()
            monthly_data = pd.concat([monthly_data, report], axis=1, sort=False)
        return results, monthly_data

    def drill_down_dfs(self, **kwargs):
        """

        Args:
            kwargs (): dictionary of dataframes that were created by COLLECT_RESULTS

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        df_dict = dict()
        for der in self.value_streams.values():
            df_dict.update(der.drill_down_reports(**kwargs))
        return df_dict
