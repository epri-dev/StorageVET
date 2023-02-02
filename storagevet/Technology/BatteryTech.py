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
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from .EnergyStorage import EnergyStorage
import numpy as np
import pandas as pd
import rainflow
from storagevet.ErrorHandling import *
from storagevet.Library import truncate_float, is_leap_yr
import cvxpy as cvx


class Battery(EnergyStorage):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, params):
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
           params (dict): params dictionary from dataframe for one case
        """
        TellUser.debug(f"Initializing {__name__}")
        self.tag = 'Battery'
        # create generic storage object
        super().__init__(params)

        self.hp = params['hp']

        self.tag = 'Battery'
        # initialize degradation attributes
        self.cycle_life = params['cycle_life']
        self.degrade_perc = 0
        self.soh_initial = 1 #Initial SOC at the start of the project
        self.soh=1 #Initial SOC at the start of the project
        self.yearly_degrade = params['yearly_degrade'] / 100
        self.eol_condition = params['cycle_life_table_eol_condition'] / 100
        self.incl_cycle_degrade = bool(params['incl_cycle_degrade'])
        self.degrade_data = None
        self.counted_cycles = []

    def initialize_degradation_module(self, opt_agg):
        """

        Notes: Should be called once, after optimization levels are assigned, but before
        optimization loop gets called

        Args:
            opt_agg (DataFrame):

        Returns: None

        """
        if self.incl_cycle_degrade:
            # initialize degradation dataframe
            self.degrade_data = pd.DataFrame(index=['Optimization Start']+list(opt_agg.control.unique()))
            self.degrade_data['degradation progress %'] = self.degrade_perc
            self.degrade_data['state of health %'] = self.soh *100
            self.degrade_data['effective energy capacity (kWh)'] = self.degraded_energy_capacity()
            self.calc_degradation('Optimization Start', None, None)

    def degraded_energy_capacity(self):
        """ Updates ene_max_rated and control constraints based on degradation percent
        Applies degrade percent to rated energy capacity

        TODO: use lookup table for energy cap to degredation percentage

        Returns:
            Degraded energy capacity
        """

        soh_change = self.degrade_perc
        new_ene_max = max(self.ene_max_rated * (1 - soh_change), 0)
        return new_ene_max

    def calc_degradation(self, opt_period, start_dttm, last_dttm):
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            opt_period: the index of the optimization that occurred before calling this function, None if
                no optimization problem has been solved yet
            start_dttm (DateTime): Start timestamp to calculate degradation. ie. the first datetime in the optimization
                problem
            last_dttm (DateTime): End timestamp to calculate degradation. ie. the last datetime in the optimization
                problem

        A percent that represented the energy capacity degradation
        """

        # time difference between time stamps converted into years multiplied by yearly degrate rate
        if self.incl_cycle_degrade:
            cycle_degrade = 0
            yearly_degradation = 0

            if not isinstance(opt_period, str):
                # calculate degradation due to cycling iff energy values are given
                energy_series = self.variables_df.loc[start_dttm:last_dttm, 'ene']
                # Find the effective energy capacity
                eff_e_cap = self.degraded_energy_capacity()

                #If using rainflow counting package uncomment following few lines
                # use rainflow counting algorithm to get cycle counts
                # cycle_counts = rainflow.count_cycles(energy_series, ndigits=4)
                #
                # aux_df = pd.DataFrame(cycle_counts, columns=['DoD', 'N_cycles'])
                # aux_df['Opt window'] = opt_period
                #
                # # sort cycle counts into user inputed cycle life bins
                # digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],[min(i[0]/eff_e_cap, 1) for i in cycle_counts], side='left')

                # use rainflow extract function to get information on each cycle
                cycle_extract=list(rainflow.extract_cycles(energy_series))
                aux_df = pd.DataFrame(cycle_extract, columns=['rng', 'mean','count','i_start','i_end'])
                aux_df['Opt window'] = opt_period

                # sort cycle counts into user inputed cycle life bins
                digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],[min(i[0] / eff_e_cap, 1) for i in cycle_extract], side='left')
                aux_df['Input_cycle_DoD_mapping'] = np.array(self.cycle_life['Cycle Depth Upper Limit'][digitized_cycles]*eff_e_cap)
                aux_df['Cycle Life Value'] = np.array(self.cycle_life['Cycle Life Value'][digitized_cycles] )

                self.counted_cycles.append(aux_df.copy())
                # sum up number of cycles for all cycle counts in each bin
                cycle_sum = self.cycle_life.loc[:, :]
                cycle_sum.loc[:, 'cycles'] = 0
                for i in range(len(cycle_extract)):
                    cycle_sum.loc[digitized_cycles[i], 'cycles'] += cycle_extract[i][2]

                # sum across bins to get total degrade percent
                # 1/cycle life value is degrade percent for each cycle
                cycle_degrade = np.dot(1/cycle_sum['Cycle Life Value'], cycle_sum.cycles)* (1 - self.eol_condition)

            if start_dttm is not None and last_dttm is not None:
                # add the yearly degradation linearly to the # of years from START_DTTM to (END_DTTM + dt)
                days_in_year = 366 if is_leap_yr(start_dttm.year) else 365
                portion_of_year = (last_dttm + pd.Timedelta(self.dt, unit='h') - start_dttm) / pd.Timedelta(days_in_year, unit='d')
                yearly_degradation = self.yearly_degrade * portion_of_year

            # add the degradation due to time passing and cycling for total degradation
            degrade_percent = cycle_degrade + yearly_degradation

            # record the degradation
            # the total degradation after optimization OPT_PERIOD must also take into account the
            # degradation that occurred before the battery was in operation (which we saved as SELF.DEGRADE_PERC)
            self.degrade_data.loc[opt_period, 'degradation progress %'] = degrade_percent + self.degrade_perc
            self.degrade_perc += degrade_percent

            soh_new = self.soh_initial - self.degrade_perc
            self.soh = self.degrade_data.loc[opt_period, 'state of health %'] = soh_new

            # apply degradation to technology (affects physical_constraints['ene_max_rated'] and control constraints)
            eff_e_cap = self.degraded_energy_capacity()
            TellUser.info(f"BATTERY - {self.name}: effective energy capacity is now {truncate_float(eff_e_cap)} kWh " +
                          f"({truncate_float(100*(1 - (self.ene_max_rated-eff_e_cap)/self.ene_max_rated), 7)}% of original)")
            self.degrade_data.loc[opt_period, 'effective energy capacity (kWh)'] = eff_e_cap
            self.effective_soe_max = eff_e_cap * self.ulsoc
            self.effective_soe_min = eff_e_cap * self.llsoc

    def constraints(self, mask, **kwargs):
        """Default build constraint list method. Used by services that do not
        have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical
                constraints and its service constraints
        """
        # create default list of constraints
        constraint_list = super().constraints(mask, **kwargs)
        if self.incl_binary:
            # battery can not charge and discharge in the same timestep
            constraint_list += [cvx.NonPos(self.variables_dict['on_c'] +
                                           self.variables_dict['on_d'] - 1)]

        return constraint_list

    def save_variable_results(self, subs_index):
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        DER instance and saves the values it to itself

        Args:
            subs_index (Index): index of the subset of data for which the variables were solved for
        """
        super().save_variable_results(subs_index)
        # check for charging and discharging in same time step
        eps = 1e-4
        if np.any((self.variables_df.loc[subs_index, 'ch'].values >= eps) & (self.variables_df.loc[subs_index, 'dis'].values >= eps)):
            TellUser.warning('non-zero charge and discharge powers found in optimization solution. Try binary formulation')

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
        if self.hp > 0:
            tech_id = self.unique_tech_id()
            # the value of the energy consumed by the auxiliary load (housekeeping power) is assumed to be equal to the
            # value of energy for DA ETS, real time ETS, or retail ETS.
            optimization_years = self.variables_df.index.year.unique()
            hp_proforma = pd.DataFrame()
            if results.columns.isin(['Energy Price ($/kWh)']).any():
                hp_cost = self.dt * -results.loc[:, 'Energy Price ($/kWh)'] * self.hp
                for year in optimization_years:
                    year_monthly = hp_cost[hp_cost.index.year == year]
                    hp_proforma.loc[pd.Period(year=year, freq='y'), tech_id + 'Aux Load Cost'] = year_monthly.sum()
            # fill forward
            hp_proforma = fill_forward_func(hp_proforma, None)
            # append will super class's proforma
            pro_forma = pd.concat([pro_forma, hp_proforma], axis=1)

        return pro_forma

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

        DCT = super().drill_down_reports(monthly_data, time_series_data, technology_summary, sizing_df)

        if self.incl_cycle_degrade:
            DCT[f"{self.name.replace(' ', '_')}_degradation_data"] = self.degrade_data

            total_counted_cycles = pd.concat(self.counted_cycles)

            DCT[f"{self.name.replace(' ', '_')}_cycle_counting"] = total_counted_cycles

        return DCT
