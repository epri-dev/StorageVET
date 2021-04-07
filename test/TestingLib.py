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
from storagevet.StorageVET import StorageVET
import os
import pandas as pd


def run_case(model_param_location: str):
    print(f"Testing {model_param_location}...")
    case = StorageVET(model_param_location, True)
    results = case.solve()
    print(results.dir_abs_path)
    return results


def check_initialization(model_param_location: str):
    print(f"Testing {model_param_location}...")
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


def assert_ran(model_param_location: str):
    results = run_case(model_param_location)
    assert_file_exists(results)


def assert_ran_with_services(model_param_location: str, services: list):
    results = run_case(model_param_location)
    assert_file_exists(results)
    value_stream_keys = results.instances[0].service_agg.value_streams.keys()
    print(set(value_stream_keys))
    assert set(services) == set(value_stream_keys)
