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
MarketServiceUpAndDown.py

This Python class contains methods and attributes that help model market
services that provide service through discharging more OR charging less
relative to the power set points.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Library as Lib


class MarketServiceUpAndDown(ValueStream):
    """ A market service that can provide services in the "up" and "down"
    directions

    """

    def __init__(self, name, full_name, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            name (str): abbreviated name
            full_name (str): the expanded name of the service
            params (Dict): input parameters
        """
        ValueStream.__init__(self, name, params)
        self.full_name = full_name
        self.combined_market = params['CombinedMarket']
        self.duration = params['duration']
        self.energy_growth = params['energyprice_growth']/100
        self.eod_avg = params['eod']
        self.eou_avg = params['eou']
        self.growth = params['growth']/100
        self.price_down = params['regd_price']
        self.price_up = params['regu_price']
        self.price_energy = params['energy_price']
        self.variable_names = {'up_ch', 'up_dis', 'down_ch', 'down_dis'}
        self.variables_df = pd.DataFrame(columns=self.variable_names)

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that
        might have slipped in. Update variable that hold timeseries data
        after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of
                loads in this simulation

        """
        self.price_energy = Lib.fill_extra_data(self.price_energy, years,
                                                self.energy_growth, frequency)
        self.price_energy = Lib.drop_extra_data(self.price_energy, years)

        self.price_up = Lib.fill_extra_data(self.price_up, years,
                                            self.growth, frequency)
        self.price_up = Lib.drop_extra_data(self.price_up, years)

        self.price_down = Lib.fill_extra_data(self.price_down, years,
                                              self.growth, frequency)
        self.price_down = Lib.drop_extra_data(self.price_down, years)

    def initialize_variables(self, size):
        """ Updates the optimization variable attribute with new optimization
        variables of size SIZE

        Variables added:
            up_ch (Variable): A cvxpy variable for freq regulation capacity to
                increase charging power
            down_ch (Variable): A cvxpy variable for freq regulation capacity to
                decrease charging power
            up_dis (Variable): A cvxpy variable for freq regulation capacity to
                increase discharging power
            down_dis (Variable): A cvxpy variable for freq regulation capacity to
                decrease discharging power

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        self.variables = {
            'up_ch': cvx.Variable(shape=size, name=f'{self.name}_up_c'),
            'down_ch': cvx.Variable(shape=size, name=f'{self.name}_regd_c'),
            'up_dis': cvx.Variable(shape=size, name=f'{self.name}_up_dis'),
            'down_dis': cvx.Variable(shape=size, name=f'{self.name}_regd_d')
        }

    def objective_function(self, mask, load_sum, tot_variable_gen,
                           generator_out_sum, net_ess_power, annuity_scalar=1):
        """ Generates the full objective function, including the optimization
        variables.

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent
                generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional
                generation within the system
            net_ess_power (list, Expression): the sum of the net power of all
                the ESS in the system. [= charge - discharge]
            annuity_scalar (float): a scalar value to be multiplied by any
                yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        Returns:
            A dictionary with the portion of the objective function that it
                affects, labeled by the expression's key. Default is {}.

        """

        # pay for reg down energy, get paid for reg up energy
        # paid revenue for capacity to do both
        size = sum(mask)

        p_regu = cvx.Parameter(size, value=self.price_up.loc[mask].values,
                               name=f'{self.name}_p_regu')
        p_regd = cvx.Parameter(size, value=self.price_down.loc[mask].values,
                               name=f'{self.name}_p_regd')
        p_ene = cvx.Parameter(size, value=self.price_energy.loc[mask].values,
                              name=f'{self.name}_price')
        eou = self.get_energy_option_up(mask)
        eod = self.get_energy_option_down(mask)
        # REGULATION DOWN: PAYMENT
        regdown_disch_payment \
            = cvx.sum(self.variables['down_dis'] * -p_regd) * annuity_scalar
        regdown_charge_payment \
            = cvx.sum(self.variables['down_ch'] * -p_regd) * annuity_scalar
        reg_down_tot = regdown_charge_payment + regdown_disch_payment

        # REGULATION UP: PAYMENT
        regup_disch_payment \
            = cvx.sum(self.variables['up_dis'] * -p_regu) * annuity_scalar
        regup_charge_payment \
            = cvx.sum(self.variables['up_ch'] * -p_regu) * annuity_scalar
        reg_up_tot = regup_charge_payment + regup_disch_payment

        # REGULATION UP & DOWN: ENERGY SETTLEMENT
        # NOTE: TODO: here we use rte_list[0] wqhich grabs the first available rte from an active ess
        #   we will want to change this to actually use all available rte values from the list
        regdown_disch_settlement \
            = cvx.sum(cvx.multiply(cvx.multiply(self.variables['down_dis'],
                                                p_ene),
                                   eod)) * self.dt * annuity_scalar
        regdown_charge_settlement \
            = cvx.sum(cvx.multiply(cvx.multiply(self.variables['down_ch'],
                                                p_ene),
                                   eod)) * self.dt * annuity_scalar / self.rte_list[0]
        e_settlement = regdown_disch_settlement + regdown_charge_settlement

        regup_disch_settlement \
            = cvx.sum(cvx.multiply(cvx.multiply(self.variables['up_dis'],
                                                -p_ene),
                                   eou)) * self.dt * annuity_scalar
        regup_charge_settlement \
            = cvx.sum(cvx.multiply(cvx.multiply(self.variables['up_ch'],
                                                -p_ene),
                                   eou)) * self.dt * annuity_scalar / self.rte_list[0]
        e_settlement += regup_disch_settlement + regup_charge_settlement

        return {f'{self.name}_regup_prof': reg_up_tot,
                f'{self.name}_regdown_prof': reg_down_tot,
                f'{self.name}_energy_settlement': e_settlement}

    def get_energy_option_up(self, mask):
        """ transform the energy option up into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.promote(self.eou_avg, mask.loc[mask].shape)

    def get_energy_option_down(self, mask):
        """ transform the energy option down into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.promote(self.eod_avg, mask.loc[mask].shape)

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                    net_ess_power, combined_rating):
        """build constraint list method for the optimization engine

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent
                generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional
                generation within the system
            net_ess_power (list, Expression): the sum of the net power of all
                the ESS in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class
                type

        Returns:
            An list of constraints for the optimization variables added to
            the system of equations
        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-self.variables['up_ch'])]
        constraint_list += [cvx.NonPos(-self.variables['down_ch'])]
        constraint_list += [cvx.NonPos(-self.variables['up_dis'])]
        constraint_list += [cvx.NonPos(-self.variables['down_dis'])]
        if self.combined_market:
            constraint_list += [
                cvx.Zero(self.variables['down_dis'] + self.variables['down_ch'] -
                         self.variables['up_dis'] - self.variables['up_ch'])
            ]

        return constraint_list

    def p_reservation_charge_up(self, mask):
        """ the amount of charging power in the up direction (supplying power
        up into the grid) that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['up_ch']

    def p_reservation_charge_down(self, mask):
        """ the amount of charging power in the up direction (pulling power
        down from the grid) that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['down_ch']

    def p_reservation_discharge_up(self, mask):
        """ the amount of charging power in the up direction (supplying power
        up into the grid) that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['up_dis']

    def p_reservation_discharge_down(self, mask):
        """ the amount of charging power in the up direction (pulling power
        down from the grid) that needs to be reserved for this value stream

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['down_dis']

    def uenergy_option_stored(self, mask):
        """ the deviation in energy due to changes in charge

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns:

        """
        eou = self.get_energy_option_up(mask)
        eod = self.get_energy_option_down(mask)
        e_ch_less = cvx.multiply(self.variables['up_ch'], eou) * self.dt
        e_ch_more = cvx.multiply(self.variables['down_ch'], eod) * self.dt
        return e_ch_less - e_ch_more

    def uenergy_option_provided(self, mask):
        """ the deviation in energy due to changes in discharge

        Args:
            mask (DataFrame): A boolean array that is true for indices
            corresponding to time_series data included in the subs data set

        Returns:

        """
        eou = self.get_energy_option_up(mask)
        eod = self.get_energy_option_down(mask)
        e_dis_less = cvx.multiply(self.variables['down_dis'], eod) * self.dt
        e_dis_more = cvx.multiply(self.variables['up_dis'], eou) * self.dt
        return e_dis_more - e_dis_less

    def worst_case_uenergy_stored(self, mask):
        """ the amount of energy, from the current SOE that needs to be
        reserved for this value stream to prevent any violates between the
        steps in time that are not catpured in our timeseries.

        Note: stored energy should be positive and provided energy should be
            negative

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: tuple (stored, provided),
            where the first value is the case where the systems would end up
            with more energy than expected and the second corresponds to the
            case where the systems would end up with less energy than expected

        """
        stored \
            = self.variables['down_ch'] * self.duration \
            + self.variables['down_dis'] * self.duration
        return stored

    def worst_case_uenergy_provided(self, mask):
        """ the amount of energy, from the current SOE that needs to be
         reserved for this value stream to prevent any violates between the
         steps in time that are not catpured in our timeseries.

        Note: stored energy should be positive and provided energy should be
        negative

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: tuple (stored, provided),
            where the first value is the case where the systems would end up
            with more energy than expected and the second corresponds to the
            case where the systems would end up with less energy than expected

        """
        provided \
            = self.variables['up_ch'] * -self.duration \
            + self.variables['up_dis'] * -self.duration
        return provided

    def timeseries_report(self):
        """ Summaries the optimization results for this Value Stream.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """

        report = pd.DataFrame(index=self.price_energy.index)
        # GIVEN
        report.loc[:, f"{self.name} Up Price ($/kW)"] \
            = self.price_up
        report.loc[:, f"{self.name} Down Price ($/kW)"] \
            = self.price_down
        report.loc[:, f"{self.name} Energy Settlement Price ($/kWh)"] = \
            self.price_energy

        # OPTIMIZATION VARIABLES
        report.loc[:, f'{self.full_name} Down (Charging) (kW)'] \
            = self.variables_df['down_ch']
        report.loc[:, f'{self.full_name} Down (Discharging) (kW)'] \
            = self.variables_df['down_dis']
        report.loc[:, f'{self.full_name} Up (Charging) (kW)'] \
            = self.variables_df['up_ch']
        report.loc[:, f'{self.full_name} Up (Discharging) (kW)'] \
            = self.variables_df['up_dis']

        # CALCULATED EXPRESSIONS (ENERGY THROUGH-PUTS)
        e_thru_down_dis = np.multiply(self.eod_avg,
                                      self.variables_df['down_dis']) * self.dt
        e_thru_down_ch = np.multiply(self.eod_avg,
                                     self.variables_df['down_ch']) * self.dt
        e_thru_up_dis = -np.multiply(self.eou_avg,
                                     self.variables_df['up_dis']) * self.dt
        e_thru_up_ch = -np.multiply(self.eou_avg,
                                    self.variables_df['up_ch']) * self.dt
        uenergy_down = e_thru_down_dis + e_thru_down_ch
        uenergy_up = e_thru_up_dis + e_thru_up_ch

        column_start = f"{self.name} Energy Throughput"
        report.loc[:, f"{column_start} (kWh)"] = uenergy_down + uenergy_up
        report.loc[:, f"{column_start} Up (Charging) (kWh)"] = e_thru_up_ch
        report.loc[:, f"{column_start} Up (Discharging) (kWh)"] = e_thru_up_dis
        report.loc[:, f"{column_start} Down (Charging) (kWh)"] = e_thru_down_ch
        report.loc[:, f"{column_start} Down (Discharging) (kWh)"] \
            = e_thru_down_dis

        return report

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            opt_years (list): list of years the optimization problem ran for
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame (of with each year in opt_year as the
        index and the corresponding value this stream provided)

        """
        proforma = super().proforma_report(opt_years, apply_inflation_rate_func,
                                           fill_forward_func, results)
        pref = self.full_name
        reg_up = \
            results.loc[:, f'{pref} Up (Charging) (kW)'] \
            + results.loc[:, f'{pref} Up (Discharging) (kW)']
        regulation_up_prof = np.multiply(reg_up, self.price_up)

        reg_down = \
            results.loc[:, f'{pref} Down (Charging) (kW)'] \
            + results.loc[:, f'{pref} Down (Discharging) (kW)']
        regulation_down_prof = np.multiply(reg_down, self.price_down)

        # NOTE: TODO: here we use rte_list[0] wqhich grabs the first available rte from an active ess
        #   we will want to change this to actually use all available rte values from the list
        energy_throughput = \
            results.loc[:, f"{self.name} Energy Throughput Down (Charging) (kWh)"] / self.rte_list[0] \
            + results.loc[:, f"{self.name} Energy Throughput Down (Discharging) (kWh)"] \
            + results.loc[:, f"{self.name} Energy Throughput Up (Charging) (kWh)"] / self.rte_list[0] \
            + results.loc[:, f"{self.name} Energy Throughput Up (Discharging) (kWh)"]
        energy_through_prof = np.multiply(energy_throughput, self.price_energy)

        # combine all potential value streams into one df for faster
        #   splicing into years
        fr_results = pd.DataFrame({'E': energy_through_prof,
                                   'RU': regulation_up_prof,
                                   'RD': regulation_down_prof},
                                  index=results.index)
        market_results_only = proforma.copy(deep=True)
        for year in opt_years:
            year_subset = fr_results[fr_results.index.year == year]
            yr_pd = pd.Period(year=year, freq='y')
            proforma.loc[yr_pd, f'{self.name} Energy Throughput'] \
                = -year_subset['E'].sum()
            market_results_only.loc[yr_pd, f'{pref} Up'] \
                = year_subset['RU'].sum()
            market_results_only.loc[yr_pd, f'{pref} Down'] \
                = year_subset['RD'].sum()
        # forward fill growth columns with inflation at their corresponding growth rates
        market_results_only = fill_forward_func(market_results_only, self.growth)
        proforma = fill_forward_func(proforma, self.energy_growth)
        # concat the two together
        proforma = pd.concat([proforma, market_results_only], axis=1)
        return proforma
