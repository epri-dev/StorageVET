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
This file tests analysis cases that ONLY contain a SINGLE BATTERY. It is
organized by value stream combination and tests a bariety of optimization
horizons, time scale sizes, and other scenario options. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
import pytest
from pathlib import Path
import numpy as np
from test.TestingLib import *

DIR = Path('./test/model_params/')


def test_da_month():
    test_file = DIR / f'000-DA_battery_month{CSV}'
    assert_ran_with_services(test_file, ['DA'])


@pytest.mark.slow
def test_da_month3():
    assert_ran(DIR / f'018-DA_battery_month_5min{CSV}')


def test_da_12hr():
    assert_ran(DIR / f'019-DA_battery_month_12hropt{CSV}')


def test_da_month5():
    assert_ran(DIR / f'023-DA_month_results_dir_label{CSV}')


def xtest_da_month_degradation_multi_yr_battery_replaced_during_optimization():
    # TODO
    assert_ran(r".\Testing\cba_validation\Model_params" +
               r"\Model_Parameters_Template_ENEA_S1_8_12_UC1_DAETS.csv")


def test_da_month_degradation_battery_replaced_during_optimization():
    assert_ran(DIR / f"010-degradation_test{CSV}")


def test_da_fr_month():
    assert_ran(DIR / f'001-DA_FR_battery_month{CSV}')


class TestDaDeferral:
    """ Day Ahead Energy Time Shift with Deferral"""
    def setup_class(self):
        self.results = run_case(DIR / f'003-DA_Deferral_battery_month{CSV}')
        self.test_proforma_df = self.results.proforma_df()

    def test_deferral_specific_results_exists(self):
        assert_file_exists(self.results, 'deferral_results')

    def test_proforma_exists(self):
        assert_file_exists(self.results, 'pro_forma')

    def test_number_of_analysis_years(self):  # should run for years: 2017 2023 2024
        analysis_year = self.results.instances[0].opt_years
        assert len(analysis_year) == 3

    def test_analysis_years_are_expected(self):  # should run for years: 2017 2023 2024
        analysis_year = self.results.instances[0].opt_years
        assert sorted(analysis_year) == sorted([2017, 2023, 2024])

    def test_proforma_values_are_not_zero(self):  # values from 2017 through 2023
        value = self.test_proforma_df.loc[self.test_proforma_df.index != 'CAPEX Year', 'Deferral Value']
        assert np.all(value[value.index <= pd.Period(2023, freq='y')] != 0)

    def test_proforma_values_inflate_from_start(self):  # expected inflation (3%)
        value = self.test_proforma_df.loc[self.test_proforma_df.index != 'CAPEX Year', 'Deferral Value']
        expected_inflation_after_max_opt_yr = [1.03**year for year in range(2023-2017+1)]
        assert np.all(value[value.index <= pd.Period(2023, freq='y')].values/value.values[0] == expected_inflation_after_max_opt_yr)


@pytest.mark.slow
def test_da_nsr_month():
    assert_ran(DIR / f'005-DA_NSR_battery_month{CSV}')


@pytest.mark.slow
def test_da_nsr_month1():
    assert_ran(DIR / f'007-nsr_battery_multiyr{CSV}')


@pytest.mark.slow
def test_da_sr_month():
    assert_ran(DIR / f'006-DA_SR_battery_month{CSV}')


def test_da_user_month():
    assert_ran(DIR / f'011-DA_User_battery_month{CSV}')


def test_da_ra_month():
    assert_ran_with_services(DIR / f'012-DA_RApeakmonth_battery_month{CSV}', ['DA', 'RA'])


def test_da_ra_month1():
    assert_ran_with_services(DIR / f'013-DA_RApeakmonthActive_battery_month{CSV}', ['DA', 'RA'])


def test_da_ra_month2():
    assert_ran_with_services(DIR / f'014-DA_RApeakyear_battery_month{CSV}', ['DA', 'RA'])


class TestDayAheadDemandResponse:
    """ Day Ahead Demand Response program model"""

    def setup_class(self):
        self.results = run_case(DIR / f'015-DA_DRdayahead_battery_month{CSV}')
        self.results_instance = self.results.instances[0]
        timeseries = self.results_instance.time_series_data
        self.discharge_constraint = timeseries.loc[:, "DR Discharge Min (kW)"]
        self.battery_discharge = timeseries.loc[:, 'BATTERY: 2mw-5hr Discharge (kW)']
        self.battery_charge = timeseries.loc[:, 'BATTERY: 2mw-5hr Charge (kW)']

    def test_services_were_part_of_problem(self):
        assert_usecase_considered_services(self.results, ['DA', 'DR'])

    def test_qualifying_commitment_calculation(self):
        dr_obj = self.results_instance.service_agg.value_streams['DR']
        assert np.all(dr_obj.qc[dr_obj.qc.index.month.isin([6,7,9])] == 10) and np.all(dr_obj.qc[dr_obj.qc.index.month==8] == 20)

    def test_number_of_events(self):  # num of events == 10  length of event == 4  dt == 1
        active_constraint_indx = self.discharge_constraint[self.discharge_constraint != 0].index
        assert len(active_constraint_indx) == 10 * 4 * 1

    def test_expected_discharge1(self):
        dr_events = self.discharge_constraint[self.discharge_constraint != 0]
        assert np.all(dr_events[dr_events.index.month.isin([6,7,9])] == 10)

    def test_expected_discharge2(self):
        dr_events = self.discharge_constraint[self.discharge_constraint != 0]
        assert np.all(dr_events[dr_events.index.month == 8] == 20)

    def test_no_events_occured_in_non_active_months(self):
        assert np.all(self.discharge_constraint[~self.discharge_constraint.index.month.isin([6,7,8,9])] == 0)

    def test_discharge_constraint_is_met(self):
        assert np.all((self.battery_discharge - self.battery_charge) >= self.discharge_constraint)

class TestDayOfDemandResponse:
    """ Day Ahead Demand Response program model"""

    def setup_class(self):
        self.results = run_case(DIR / f'016-DA_DRdayof_battery_month{CSV}')
        self.results_instance = self.results.instances[0]
        self.timeseries = self.results_instance.time_series_data
        self.discharge_constraint = self.timeseries.loc[:, "DR Possible Event (y/n)"]

    def test_services_were_part_of_problem(self):
        assert_usecase_considered_services(self.results, ['DA', 'DR'])

    def test_qualifying_energy_calculation(self):
        dr_obj = self.results_instance.service_agg.value_streams['DR']
        assert np.all(dr_obj.qe[dr_obj.qe.index.month.isin([6,7,9])] == 40) and np.all(dr_obj.qe[dr_obj.qe.index.month == 8] == 80)

    def test_events_might_occur_when_expected(self):  # active months = 6,7,8,9  active hours = (11 he,14 he) not on weekends
        active_constraint_indx = self.discharge_constraint[self.discharge_constraint].index
        assert np.all(active_constraint_indx.month.isin([6,7,8,9]) & active_constraint_indx.hour.isin([10,11,12,13,14]) & (active_constraint_indx.weekday < 5))

    def test_enough_energy_at_start_of_potential_event1(self):  # length of event == 4  last potential start == 14-4=10
        battery_state_of_energy = self.timeseries.loc[:, "BATTERY: 2mw-5hr State of Energy (kWh)"]
        soe_start = battery_state_of_energy[(self.discharge_constraint.values) & (self.discharge_constraint.index.hour == 10)]
        assert np.all(soe_start[soe_start.index.month.isin([6,7,9])] >= 40)

    def test_enough_energy_at_start_of_potential_event2(self):  # length of event == 4  last potential start == 14-4=10
        battery_state_of_energy = self.timeseries.loc[:, "BATTERY: 2mw-5hr State of Energy (kWh)"]
        soe_start = battery_state_of_energy[(self.discharge_constraint.values) & (self.discharge_constraint.index.hour == 10)]
        assert np.all(soe_start[soe_start.index.month == 8] >= 80)

class TestEnergyConstraints:
    """ Multiple System Constraints on Energy: Backup Energy (a Monthyl Min Constraint)
            and User Constraints on Energy (Min and Max)
    """

    def setup_class(self):
        self.results = run_case(DIR / f'055-DA_Battery_Backup_User{CSV}')
        self.results_instance = self.results.instances[0]
        timeseries = self.results_instance.time_series_data
        self.backup = timeseries.loc[:, "Backup Energy Reserved (kWh)"]
        self.soe = timeseries.loc[:, 'BATTERY: ess1 State of Energy (kWh)']
        self.agg_soe = timeseries.loc[:, 'Aggregated State of Energy (kWh)']
        self.user_energy_max = timeseries.loc[:, 'User Constraints Aggregate Energy Max (kWh)']
        self.user_energy_min = timeseries.loc[:, 'User Constraints Aggregate Energy Min (kWh)']

    def test_services_were_part_of_problem(self):
        assert_usecase_considered_services(self.results, ['DA', 'Backup', 'User'])

    def test_max_energy_constraint_is_met(self):
        assert np.all(self.soe <= self.user_energy_max)
        assert np.all(self.agg_soe <= self.user_energy_max)

    def test_min_energy_constraint_is_met(self):
        assert np.all(self.soe >= self.user_energy_min)
        assert np.all(self.agg_soe >= self.user_energy_min)

    def test_backup_energy_constraint_is_met(self):
        assert np.all(self.soe >= self.backup)
        assert np.all(self.agg_soe >= self.backup)
