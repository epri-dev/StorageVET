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
This file tests bill reduction analysis cases. All tests should pass.

The tests in this file can be run with DERVET and StorageVET, so make sure to
update TEST_PROGRAM with the lower case string name of the program that you
would like the tests to run on.

"""
from test.TestingLib import assert_ran, run_case
from pathlib import Path


DIR = Path('./test/model_params/')


def test_ice():
    assert_ran(DIR / "030-billreduction_ice_month.csv")


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
