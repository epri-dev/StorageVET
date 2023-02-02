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
This file tests features of the CBA module. All tests should pass.

The tests in this file can be run with .

"""
import pytest
from test.TestingLib import assert_ran, run_case
from pathlib import Path
from storagevet.ErrorHandling import *
import pandas as pd
import numpy as np

DIR = Path("./test/model_params")

class TestProformaWithDegradation:
    """
    Test Proforma: degradation, retailETS growth rate = 0, inflation rate = 3%,
    no fixed or variable OM costs
    """
    def setup_class(self):
        # run case
        run_results = run_case(DIR / "040-Degradation_Test_MP.csv")
        # get proforma
        self.actual_proforma = run_results.proforma_df()
        self.energy_charges = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                                       'Avoided Energy Charge']

    def test_all_project_years_are_in_proforma(self):
        expected_index = pd.period_range(2017, 2030, freq='y')
        expected_index = set(expected_index.values)
        expected_index.add('CAPEX Year')
        assert set(self.actual_proforma.index.values) == expected_index

    def test_years_btw_and_after_optimization_years_are_filed(self):
        assert np.all(self.actual_proforma['Yearly Net Value'])

    def test_older_opt_year_energy_charge_values_less(self):
        assert self.energy_charges[pd.Period(2017, freq='y')] > self.energy_charges[pd.Period(
            2022, freq='y')]

    def test_non_opt_year_energy_charge_values_same_as_last_opt_year(self):
        last_opt_year = pd.Period(2022, freq='y')
        assert np.all((self.energy_charges[self.energy_charges.index > last_opt_year] /
                       self.energy_charges[last_opt_year]) == 1)


class TestProformaWithNoDegradation:
    """
    Test Proforma: no degradation, retailETS growth rate = 0%, inflation rate = 3%,
    non-zero fixed and variable OM costs
    """
    def setup_class(self):
        # run case
        run_results = run_case(DIR / "041-no_Degradation_Test_MP.csv")
        # get proforma
        self.actual_proforma = run_results.proforma_df()
        self.energy_charges = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                                       'Avoided Energy Charge']
        self.inflation_rate = [1.03**(year.year - 2017) for year in self.energy_charges.index]

    def test_opt_year_energy_charge_values_same(self):
        # growth rate = 0, so all opt year (2017, 2022) values should be the same
        assert self.energy_charges[pd.Period(2017, freq='y')] == self.energy_charges[pd.Period(2022, freq='y')]

    def test_non_opt_year_energy_charge_values(self):
        assert np.all((self.energy_charges / self.energy_charges[pd.Period(2017, freq='y')]) == 1)

    def test_variable_om_values_reflect_inflation_rate(self):
        variable_om = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                               'BATTERY: es Variable O&M Cost'].values
        deflated_cost = variable_om / self.inflation_rate
        compare_cost_to_base_year_value = list(deflated_cost / deflated_cost[0])
        assert np.allclose(compare_cost_to_base_year_value, list(np.ones(len(variable_om))))

    def test_fixed_om_values_reflect_inflation_rate(self):
        fixed_om = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                            'BATTERY: es Fixed O&M Cost'].values
        deflated_cost = fixed_om / self.inflation_rate
        compare_cost_to_base_year_value = list(deflated_cost / deflated_cost[0])
        assert np.allclose(compare_cost_to_base_year_value, list(np.ones(len(fixed_om))))


class TestProformaWithNoDegradationNegRetailGrowth:
    """
    Test Proforma: no degradation, retailETS growth rate = -10%, inflation rate = 3%,
    non-zero fixed and variable OM costs
    """
    def setup_class(self):
        # run case
        run_results = run_case(DIR / "042-no_Degradation_Test_MP_tariff_neg_grow_rate.csv")
        # get proforma
        self.actual_proforma = run_results.proforma_df()
        self.energy_charges = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                                       'Avoided Energy Charge']
        self.inflation_rate = [1.03 ** (year.year - 2017) for year in self.energy_charges.index]

    def test_opt_year_energy_charge_values_should_reflect_growth_rate(self):
        # growth rate = 0, so all opt year (2017, 2022) values should be the same
        assert self.energy_charges[pd.Period(2017, freq='y')] > self.energy_charges[pd.Period(2022, freq='y')]

    def test_years_beyond_max_opt_year_energy_charge_values_reflect_growth_rate(self):
        years_beyond_max = self.energy_charges[pd.Period(2023, freq='y'):]
        max_opt_year_value = self.energy_charges[pd.Period(2022, freq='y')]
        charge_growth_rate = [.9 ** (year.year-2022) for year in years_beyond_max.index]
        expected_values = years_beyond_max / charge_growth_rate
        assert np.allclose(expected_values.values, max_opt_year_value)

    def test_variable_om_values_reflect_inflation_rate(self):
        variable_om = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                               'BATTERY: es Variable O&M Cost'].values
        deflated_cost = variable_om / self.inflation_rate
        compare_cost_to_base_year_value = list(deflated_cost / deflated_cost[0])
        assert np.allclose(compare_cost_to_base_year_value, list(np.ones(len(variable_om))))

    def test_fixed_om_values_reflect_inflation_rate(self):
        fixed_om = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year',
                                            'BATTERY: es Fixed O&M Cost'].values
        deflated_cost = fixed_om / self.inflation_rate
        compare_cost_to_base_year_value = list(deflated_cost / deflated_cost[0])
        assert np.allclose(compare_cost_to_base_year_value, list(np.ones(len(fixed_om))))


def test_years_beyond_max_opt_year_da_energy_charges_values_reflect_growth_rate():
    run_results = run_case(DIR / '043-DA_battery_month_multiyear_proj.csv')
    # get proforma
    actual_proforma = run_results.proforma_df()
    energy_charges = actual_proforma.loc[actual_proforma.index != 'CAPEX Year', 'DA ETS']
    years_beyond_max = energy_charges[pd.Period(2018, freq='y'):]
    max_opt_year_value = energy_charges[pd.Period(2017, freq='y')]
    charge_growth_rate = [(1.09) ** (year.year - 2017) for year in years_beyond_max.index]
    expected_values = years_beyond_max / charge_growth_rate
    assert np.allclose(expected_values.values, max_opt_year_value)

class TestProformaWithNoDegradationNegDAGrowth:
    """
    Test Proforma: no degradation, DA growth rate = -50%, inflation rate = 3%,
    non-zero fixed and variable OM costs, DA energy prices vary between
    2 optimization years (and thus cumulative energy dispatch differs)
    """
    def setup_class(self):
        # run case
        run_results = run_case(DIR / "054-no_Degradation_Test_MP_tariff_neg_grow_rate_da.csv")
        # get proforma
        self.actual_proforma = run_results.proforma_df()

        # get year data
        self.start_year = run_results.instances[0].start_year.year
        self.end_year = run_results.instances[0].end_year.year
        self.opt_years = run_results.instances[0].opt_years
        self.min_opt_year = min(self.opt_years)
        self.max_opt_year = max(self.opt_years)
        self.dt = run_results.instances[0].dt

        # get time series output
        self.time_series_out = run_results.instances[0].time_series_data

        # get params
        self.fixed_om = 12
        self.variable_om = 0.5
        self.dis_max_rated = 100
        self.inflation_rate = 1.03


    def test_verify_variable_om_costs(self):
        ts_dis = self.time_series_out.loc[:, 'BATTERY: es Discharge (kW)']
        dis_sum_opt_year1 = np.sum(ts_dis.loc[ts_dis.index.year == self.opt_years[0]])
        dis_sum_opt_year2 = np.sum(ts_dis.loc[ts_dis.index.year == self.opt_years[1]])
        actual_om_cost_series = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year', 'BATTERY: es Variable O&M Cost']
        om_cost_dict = {}
        # NOTE: this test is only valid for 2 opt years
        for y in range(self.start_year, self.end_year + 1):
            if y in self.opt_years:
                # on an opt year
                dis_sum = np.sum(ts_dis.loc[ts_dis.index.year == y])
            elif (y > self.min_opt_year and y < self.max_opt_year):
                # between opt years
                dis_sum = dis_sum_opt_year1 - (dis_sum_opt_year1 - dis_sum_opt_year2) * (y - self.min_opt_year) / (self.max_opt_year - self.min_opt_year)
            elif y < self.min_opt_year:
                # prior to opt years
                dis_sum = dis_sum_opt_year1
            elif y > self.max_opt_year:
                # after opt years
                dis_sum = dis_sum_opt_year2

            print(y, dis_sum)
            om_cost_dict[y] = (-self.variable_om * 1e-3 * self.inflation_rate ** (y - self.min_opt_year)) * (dis_sum)
        om_cost_series = pd.Series(om_cost_dict)
        om_cost_series.name = 'BATTERY: es Variable O&M Cost'
        om_cost_series.index.name = 'Year'
        print(om_cost_series)
        print(actual_om_cost_series)
        assert np.allclose(actual_om_cost_series, om_cost_series)


    def test_verify_fixed_om_costs(self):
        actual_om_cost_series = self.actual_proforma.loc[self.actual_proforma.index != 'CAPEX Year', 'BATTERY: es Fixed O&M Cost']
        om_cost_dict = {}
        for y in range(self.start_year, self.end_year + 1):
            om_cost_dict[y] = -self.dis_max_rated * self.fixed_om * self.inflation_rate ** (y - self.min_opt_year)
        om_cost_series = pd.Series(om_cost_dict)
        om_cost_series.name = 'BATTERY: es Fixed O&M Cost'
        om_cost_series.index.name = 'Year'
        print(om_cost_series)
        print(actual_om_cost_series)
        assert np.allclose(actual_om_cost_series, om_cost_series)
