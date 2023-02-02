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
from storagevet.StorageVET import StorageVET
import os
import pandas as pd
import numpy as np
from pathlib import Path

DIR = Path("./")
JSON = '.json'
CSV = '.csv'

DEFAULT_MP = DIR / f'Model_Parameters_Template'
TEMP_MP = DIR / f'temp_model_parameters'


def _checkThatFileExists(f, name='Unlabeled', raise_exception_on_fail=True, write_msg_to_terminal=True):
    path_file = Path(f)
    if write_msg_to_terminal:
        msg = f'\n{name} file:\n  {path_file.resolve()}'
        print(msg)
    if not path_file.is_file():
        if raise_exception_on_fail:
            raise FileNotFoundError(f'\n\nFAIL: Your specified {name} file does not exist:\n{path_file.resolve()}\n')
        else:
            print(f'\n\nFAIL: Your specified {name} file does not exist:\n{path_file.resolve()}\n')
            return None
    return path_file

def run_case(model_param_location: str):
    print(f"Testing {model_param_location}...")
    # first make sure the model_param file exists
    model_param_file = _checkThatFileExists(Path(model_param_location), 'Model Parameter Input File')
    case = StorageVET(model_param_location, True)
    results = case.solve()
    print(results.dir_abs_path)
    return results

def check_initialization(model_param_location: str):
    print(f"Testing {model_param_location}...")
    # first make sure the model_param file exists
    model_param_file = _checkThatFileExists(Path(model_param_location), 'Model Parameter Input File')
    case = StorageVET(model_param_location, True)
    return case

def assert_file_exists(model_results, results_file_name='timeseries_results'):
    if model_results.sensitivity_df.empty:
        check_for_file = model_results.dir_abs_path / f'{results_file_name}{model_results.csv_label}.csv'
        assert os.path.isfile(check_for_file), f'No {results_file_name} found at {check_for_file}'
    else:
        for index in model_results.instances.keys():
            check_for_file = model_results.dir_abs_path / str(index) / f'{results_file_name}{model_results.csv_label}.csv'
            assert os.path.isfile(check_for_file), f'No {results_file_name} found at {check_for_file}'

def assert_file_does_not_exist(model_results, results_file_name: str):
    if model_results.sensitivity_df.empty:
        check_for_file = model_results.dir_abs_path / f'{results_file_name}{model_results.csv_label}.csv'
        assert os.path.isfile(check_for_file) == False, f'{results_file_name} found at {check_for_file}, but no file was expected'
    else:
        for index in model_results.instances.keys():
            check_for_file = model_results.dir_abs_path / str(index) / f'{results_file_name}{model_results.csv_label}.csv'
            assert os.path.isfile(check_for_file) == False, f'{results_file_name} found at {check_for_file} but no file was expected'

def assert_within_error_bound(expected_value: float, actual_value: float, error_bound: float, error_message: str):
    if error_bound < 0:
        raise Exception(f'Testing Fail: the specified error_bound ({error_bound}) must be a positive number')
    diff = abs(expected_value-actual_value)
    if not diff:
        # the difference in values is zero
        return
    assert diff/abs(expected_value) <= error_bound/100, f'Test value: {actual_value}   Should be in range: ({expected_value+(expected_value*(error_bound/100))},{expected_value-(expected_value*(error_bound/100))}). {error_message}'


##########################################


def assert_ran(model_param_location: str):
    results = run_case(model_param_location)
    assert_file_exists(results)
    return results

def assert_ran_with_services(model_param_location: str, services: list):
    results = run_case(model_param_location)
    assert_file_exists(results)
    value_stream_keys = results.instances[0].service_agg.value_streams.keys()
    print(set(value_stream_keys))
    assert set(services) == set(value_stream_keys)

def assert_usecase_considered_services(results, services: list):
    value_stream_keys = results.instances[0].service_agg.value_streams.keys()
    print(set(value_stream_keys))
    assert set(services) == set(value_stream_keys)

def compare_proforma_results(results, frozen_proforma_location: str, error_bound: float, opt_years=None):
    if isinstance(results, pd.DataFrame):
        actual_proforma_df = frozen_proforma_location
    else:
        assert_file_exists(results, 'pro_forma') # assert that pro_forma.csv file exists
        actual_proforma_df = results.proforma_df()
    if isinstance(frozen_proforma_location, pd.DataFrame):
        expected_df = frozen_proforma_location
    else:
        try:
            expected_df = pd.read_csv(frozen_proforma_location, index_col='Unnamed: 0')
        except ValueError:
            expected_df = pd.read_csv(frozen_proforma_location, index_col='Year')
    for yr_indx, values_series in expected_df.iterrows():
        print(f'\nPROFORMA YEAR: {yr_indx}\n')
        try:
            actual_indx = pd.Period(yr_indx)
            if opt_years is not None and actual_indx.year not in opt_years:
                continue
        except ValueError:
            actual_indx = yr_indx
        assert actual_indx in actual_proforma_df.index, f'{actual_indx} not in test proforma index'
        for col_indx in values_series.index:
            # NOTE: this loops through expected columns (extra columns appearing in the actual
            #       proforma file are ignored)
            print(col_indx)
            assert col_indx in actual_proforma_df.columns, f'{col_indx} not in test proforma columns'
            error_message = f'ValueError in Proforma [{yr_indx}, {col_indx}]\n'
            print(expected_df.loc[yr_indx, col_indx], actual_proforma_df.loc[actual_indx, col_indx])
            assert_within_error_bound(expected_df.loc[yr_indx, col_indx], actual_proforma_df.loc[actual_indx, col_indx], error_bound, error_message)

def compare_npv_results(results, frozen_npv_location: str, error_bound: float, opt_years=None):
    if isinstance(results, pd.DataFrame):
        actual_npv_df = results
    else:
        assert_file_exists(results, 'npv')  # assert that npv.csv file exists
        actual_npv_df = results.instances[0].cost_benefit_analysis.npv
    if isinstance(frozen_npv_location, pd.DataFrame):
        expected_df = frozen_npv_location
    else:
        try:
            expected_df = pd.read_csv(frozen_npv_location, index_col='Unnamed: 0')
        except ValueError:
            expected_df = pd.read_csv(frozen_npv_location, index_col='Year')
    for yr_indx, values_series in expected_df.iterrows():
        print(f'\n{yr_indx}:\n')
        try:
            actual_indx = pd.Period(yr_indx)
            if opt_years is not None and actual_indx.year not in opt_years:
                continue
        except ValueError:
            actual_indx = yr_indx
        assert actual_indx in actual_npv_df.index, f'{actual_indx} not in test npv index'
        for col_indx in values_series.index:
            # NOTE: this loops through expected columns (extra columns appearing in the actual
            #       npv file are ignored)
            print(col_indx)
            assert col_indx in actual_npv_df.columns, f'{col_indx} not in test npv columns'
            error_message = f'ValueError in NPV [{yr_indx}, {col_indx}]\n'
            print(expected_df.loc[yr_indx, col_indx], actual_npv_df.loc[actual_indx, col_indx])
            assert_within_error_bound(expected_df.loc[yr_indx, col_indx], actual_npv_df.loc[actual_indx, col_indx], error_bound, error_message)

def modify_mp(tag, key='name', value='yes', column='Active', mp_in=DEFAULT_MP, mp_out_tag=None):
    # read in default MP, modify it, write it to a temp file
    mp = pd.read_csv(f'{mp_in}{CSV}')
    indexes = (mp.Tag == tag) & (mp.Key == key)
    indexes = indexes[indexes].index.values
    if len(indexes) != 1:
        raise Exception(f'a unique row from the default model parameters cannot be determined (tag: {tag}, key: {key}')
    mp_cell = (indexes[0], column)
    mp.loc[mp_cell] = value
    if mp_out_tag is None:
        tempfile_name = f'{TEMP_MP}--{tag}'
    else:
        tempfile_name = f'{TEMP_MP}--{mp_out_tag}'
    mp.to_csv(f'{tempfile_name}{CSV}', index=False)
    return tempfile_name

def remove_temp_files(temp_mp):
    Path(f'{temp_mp}{CSV}').unlink()
    Path(f'{temp_mp}{JSON}').unlink()
