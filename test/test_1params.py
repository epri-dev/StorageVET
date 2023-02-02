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
This file tests features of the params class. All tests should pass.

The tests in this file can be run with .

"""
import pytest
from pathlib import Path
from test.TestingLib import *
from storagevet.ErrorHandling import *
from test.test_default_model_parameters import *


DIR = Path("./test/model_params")

"""
Timestep frequency checks
"""

def test_invalid_dt():
    # invalid dt
    # loop through invalid dt values
    # should raise error for each dt_val in the loop
    for dt_val in [0.01, 0.08, 0.15, 0.2, 0.66, 2, -1]:
        temp_mp = modify_mp('Scenario', key='dt', value=dt_val, column='Value')
        with pytest.raises(ModelParameterError):
            setup_default_case(temp_mp)
    remove_temp_files(temp_mp)


def test_freq_mismatch_dt_timeseries():
    # timestep freq mismatch between dt and length of timeseries data
    # loop through valid dt values that are not 1-hour
    # should raise error for each dt_val in the loop
    for dt_val in [0.5, 0.25, 0.166, 0.167, 0.083, 0.016, 0.0167, 0.016666666, 0.0166666667]:
        print(f'testing when dt = {dt_val}')
        temp_mp = modify_mp('Scenario', key='dt', value=dt_val, column='Value')
        with pytest.raises(TimeseriesDataError):
            setup_default_case(temp_mp)
    remove_temp_files(temp_mp)


"""
Tariff File checks
"""


def test_missing_tariff_row():
    # following should fail
    with pytest.raises(TariffError):
        check_initialization(DIR/'002-missing_tariff.csv')


def test_single_tariff_row():
    # following should pass
    run_case(DIR/'051-tariff-single_billing_period_ok.csv')


def test_multi_tariff_row():
    # following should pass
    run_case(DIR/'052-tariff-multi_billing_periods_ok.csv')


def test_repeated_tariff_billing_period():
    # following should fail
    with pytest.raises(TariffError):
        check_initialization(DIR/'053-tariff-repeated-billing-period-index.csv')


"""
Sensitivity checks
"""


def test_number_of_cases_in_sensitivity_analysis():
    model_param_location = DIR/'009-bat_energy_sensitivity.csv'
    results = run_case(model_param_location)
    assert_file_exists(results)
    assert len(results.instances.keys()) == 4


def test_number_of_cases_in_coupling():
    model_param_location = DIR/'017-bat_timeseries_dt_sensitivity_couples.csv'
    results = run_case(model_param_location)
    assert_file_exists(results)
    assert len(results.instances.keys()) == 2


def test_coupled_with_nonexisting_input_error():
    # following should fail
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/'020-coupled_dt_timseries_error.csv')


"""
DR parameter checks
"""


def test_dr_length_nan_allowed():
    """ Test if DR allows length to be nan
    """
    check_initialization(DIR/"022-DR_length_nan.csv")


def test_dr_program_end_hour_nan_allowed():
    """ Test if DR allows program_end_hour to be nan
    """
    check_initialization(DIR/"021-DR_program_end_nan.csv")


def test_dr_length_empty_allowed():
    """ Test if DR allows length to be empty
    """
    check_initialization(DIR/"047-DR_length_empty.csv")


def test_dr_program_end_hour_empty_allowed():
    """ Test if DR allows program_end_hour to be empty
    """
    check_initialization(DIR/"046-DR_program_end_empty.csv")


def test_dr_length_and_program_end_ok():
    """ Test that DR allows length and program end
        - as long as they are compatible
    """
    check_initialization(DIR/"044-DR_program_end_with_length_ok.csv")


def test_dr_length_and_program_end_error():
    """ Test that DR does not allow length and program end
        - if they are not compatible
    """
    with pytest.raises(ModelParameterError):
        run_case(DIR/"045-DR_program_end_with_length_error.csv")


def test_dr_two_nans_not_allowed():
    """ Test if DR allows both length and program end to be nan
    """
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/"024-DR_nan_length_prgramd_end_hour.csv")


def test_dr_two_empties_not_allowed():
    """ Test if DR allows both length and program end to be empty
    """
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/"048-DR_empty_length_prgramd_end_hour.csv")


def test_dr_length_unsupported_value_types():
    """ Test that DR does not allow length to be a non-nan string
    """
    with pytest.raises(ModelParameterError):
        run_case(DIR/"049-DR_unsupported_length.csv")


def test_dr_end_hour_unsupported_value_types():
    """ Test that DR does not allow program_end_hour to be a non-nan string
    """
    with pytest.raises(ModelParameterError):
        run_case(DIR/"050-DR_unsupported_program_end_hour.csv")


"""
Test opt_year checks on referenced file data
"""


def test_opt_years_not_in_timeseries_data():
    """ Test if opt_year not matching the data in timeseries file is caught
    """
    with pytest.raises(TimeseriesDataError):
        check_initialization(DIR / "025-opt_year_more_than_timeseries_data.csv")


def test_continuous_opt_years_in_timeseries_data():
    """ Test if opt_year matching the data in timeseries file is cleared. Opt_years are continuous.
    """
    assert_ran(DIR / "038-mutli_opt_years_continuous.csv")


def test_discontinuous_opt_years_in_timeseries_data():
    """ Test if opt_year matching the data in timeseries file is cleared. Opt_years are not
    continuous
    """
    assert_ran(DIR / "037-mutli_opt_years_discontinuous.csv")


def test_opt_years_not_in_monthly_data():
    """ Test if opt_year not matching the data in monthly file is caught
    """
    with pytest.raises(MonthlyDataError):
        check_initialization(DIR / "039-mutli_opt_years_not_in_monthly_data.csv")


"""
Test fuel cost with ICE
"""

def test_unallowed_fuel_type():
    # following should fail
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/'043-fuelcost-bad.csv')

def test_fuel_cost_function():
    """ Test that ICE fuel_cost equals fuel_price_liquid set in Finance.
        (ICE fuel_type is liquid)
    """
    results = run_case(DIR/'044-fuelcost.csv')
    # these 2 variables are set in the Model Params file
    ice_name = 'ice gen'
    finance_fuel_price_liquid = 25.0
    actual_fuel_costs = {}
    for der in results.instances[0].poi.der_list:
        try:
            actual_fuel_costs[der.name] = der.fuel_cost
        except(AttributeError):
            pass
    expected_fuel_costs = { ice_name: finance_fuel_price_liquid }
    assert expected_fuel_costs == actual_fuel_costs


def test_no_label_results_key():
    """ Test if opt_year matching the data in timeseries file is cleared. Opt_years are not
    continuous
    """
    assert_ran(DIR / "042-no_results_label.csv")
