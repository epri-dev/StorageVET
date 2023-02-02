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
import pandas
from storagevet.ErrorHandling import *
from test.TestingLib import *


def setup_default_case(test_file):
    case = check_initialization(f'{test_file}{CSV}')

def timeseries_data_error(test_file):
    with pytest.raises(TimeseriesDataError):
        assert_ran(f'{test_file}{CSV}')

def timeseries_missing_error(test_file):
    with pytest.raises(TimeseriesMissingError):
        assert_ran(f'{test_file}{CSV}')

def run_default_case(test_file):
    assert_ran(f'{test_file}{CSV}')

def remove_temp_files(temp_mp):
    Path(f'{temp_mp}{CSV}').unlink()
    Path(f'{temp_mp}{JSON}').unlink()

def test_default_asis():
    setup_default_case(DEFAULT_MP)
    run_default_case(DEFAULT_MP)

def test_default_ice_active():
    temp_mp = modify_mp('ICE')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_pv_active():
    temp_mp = modify_mp('PV')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_caes_active():
    temp_mp = modify_mp('CAES')
    # turn OFF Battery (Storagevet cannot handle both CAES and Battery together)
    temp_mp = modify_mp('Battery', value='no', mp_in=temp_mp, mp_out_tag='CAES')
    setup_default_case(temp_mp)
    run_default_case(temp_mp)
    remove_temp_files(temp_mp)

def test_default_pv_active_missing_pv_timeseries():
    temp_mp = modify_mp('PV')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/pv_time_series_no_pv.csv', column='Value', mp_in=temp_mp, mp_out_tag='PV')
    timeseries_missing_error(temp_mp)
    remove_temp_files(temp_mp)

def test_default_pv_active_bad_pv_data1():
    temp_mp = modify_mp('PV')
    temp_mp = modify_mp('Scenario', key='time_series_filename', value='./test/datasets/pv_time_series_bad_pv_data1.csv', column='Value', mp_in=temp_mp, mp_out_tag='PV')
    timeseries_data_error(temp_mp)
    remove_temp_files(temp_mp)
