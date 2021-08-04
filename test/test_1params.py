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
This file tests features of the params class. All tests should pass.

The tests in this file can be run with .

"""
import pytest
from pathlib import Path
from test.TestingLib import *
from storagevet.ErrorHandling import *


DIR = Path("./test/model_params")


def test_missing_tariff_row():
    # following should fail
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/'002-missing_tariff.csv')


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


def test_dr_nan_allowed():
    """ Test if DR allows length DR program end to be defined
        - the other is allowed to be 'nan'
    """
    check_initialization(DIR/"022-DR_length_nan.csv")
    check_initialization(DIR/"021-DR_program_end_nan.csv")


def test_dr_two_nans_not_allowed():
    """ Test if DR allows length DR program end to be defined
        - the other is allowed to be 'nan'
    """
    with pytest.raises(ModelParameterError):
        check_initialization(DIR/"024-DR_nan_length_prgramd_end_hour.csv")


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
