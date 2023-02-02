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
This file tests bill reduction analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
from test.TestingLib import *
from pathlib import Path

MAX_PERCENT_ERROR = 1e-9

DIR = Path('./test/model_params/')


def poi_constraint_case(test_file):
    results_instance = assert_ran(f'{test_file}{CSV}')
    results = results_instance.instances[0]
    ts = results.time_series_data
    npv = results.cost_benefit_analysis.npv
    pro_forma = results.proforma_df()
    return ts, npv, pro_forma


def test_poi_constraints_no_effect():
    # these 2 runs should be identical
    # the 030 mp file has huge value POI constraints, so when they are turned ON, they do nothing (no effect)
    test_file1 = DIR / '030-billreduction_ice_month'
    ts1, npv1, pro_forma1 = poi_constraint_case(test_file1)
    # turn on POI constraints boolean in temporary mp
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Value', mp_in=test_file1, mp_out_tag='POI')
    ts2, npv2, pro_forma2 = poi_constraint_case(temp_mp)
    remove_temp_files(temp_mp)
    # compare results from both runs to ensure they are equal
    compare_npv_results(npv1, npv2, MAX_PERCENT_ERROR)
    compare_proforma_results(pro_forma1, pro_forma2, MAX_PERCENT_ERROR)
    ts_col = 'Net Load (kW)'
    assert np.allclose(ts1[ts_col], ts2[ts_col])


def test_poi_constraints_max_export():
    # modify 030 case so that ICE generates a lot of power (over-generation).
    #   this is seen as the TS Net Load being large and negative
    # we will constrain over-generation to a smaller value, and test that it is not exceeded
    test_file = DIR / '030-billreduction_ice_month'
    max_export = 750
    # turn on POI constraints boolean in temporary mp
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Value', mp_in=test_file, mp_out_tag='POI1')
    # set max_export
    temp_mp = modify_mp('Scenario', key='max_export', value=max_export, column='Value', mp_in=temp_mp, mp_out_tag='POI1')
    # make ICE cheaper by changing the efficiency value from 19 to 3 (less fuel is needed with a lower efficiency value)
    temp_mp = modify_mp('ICE', key='efficiency', value=3, column='Value', mp_in=temp_mp, mp_out_tag='POI1')
    # run case
    ts, npv, pro_forma = poi_constraint_case(temp_mp)
    remove_temp_files(temp_mp)

    # test results
    ts_col = 'Net Load (kW)'
    # Net Load should never be less than -max_export
    #   and since we know the case would otherwise have massive over-generation (without a max_export constraint)
    #   we can say that the minimum should equal -max_export
    error_msg = f'Net Load should never be less than -max_export (-{max_export})'
    assert_within_error_bound(-1*max_export, min(ts[ts_col]), MAX_PERCENT_ERROR, error_msg)


def test_poi_constraints_max_import():
    # the 030 case as-is will not over-generate ICE
    #   the net load is positive and ranges from 200 to 1600
    # we will constrain the net load to a smaller value, and test that it is not exceeded
    test_file = DIR / '030-billreduction_ice_month'
    max_import = -500
    # turn on POI constraints boolean in temporary mp
    temp_mp = modify_mp('Scenario', key='apply_interconnection_constraints', value=1, column='Value', mp_in=test_file, mp_out_tag='POI2')
    # set max_import
    temp_mp = modify_mp('Scenario', key='max_import', value=max_import, column='Value', mp_in=temp_mp, mp_out_tag='POI2')
    # run case
    ts, npv, pro_forma = poi_constraint_case(temp_mp)
    remove_temp_files(temp_mp)

    # test results
    ts_col = 'Net Load (kW)'
    # Net Load should not exceed -max_import
    #   and since we know the case would otherwise have a higher net load,
    #   we can say that the maximum should equal -max_import
    error_msg = f'Net Load should never exceed -max_import ({-1*max_import})'
    assert_within_error_bound(-1*max_import, max(ts[ts_col]), MAX_PERCENT_ERROR, error_msg)


def test_controllable_load_month():
    assert_ran(DIR / "031-billreduction_battery_controllableload_month.csv")


def test_ice_pv():
    assert_ran(DIR / "032-pv_ice_bill_reduction.csv")


def test_ice_ice():
    assert_ran(DIR / "033-ice_ice_bill_reduction.csv")


def test_pv_ice_ice():
    assert_ran(DIR / "034-pv_ice_ice_bill_reduction.csv")


def test_pv_pv_ice():
    assert_ran(DIR / "035-pv_pv_ice_bill_reduction.csv")


def test_battery():
    """tests fixed size with retail ETS and DCM services through 1 battery"""
    assert_ran(DIR / "004-fixed_size_battery_retailets_dcm.csv")


# def test_pv():
#     assert_ran(DIR / "036-pv_bill_reduction.csv")
