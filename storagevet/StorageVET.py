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
runStorageVET.py

This Python script serves as the initial launch point executing the Python-based version of StorageVET
(AKA StorageVET 2.0 or SVETpy).
"""

from storagevet.Scenario import Scenario
from storagevet.Params import Params
from storagevet.Result import Result
import time
from storagevet.Visualization import Visualization
from storagevet.ErrorHandling import *


class StorageVET:
    """ StorageVET API. This will eventually allow StorageVET to be imported and used like any
    other python library.

    """

    def __init__(self, model_parameters_path, verbose=False):
        """ Constructor to initialize the parameters and data needed to run StorageVET.
        Initialize the Params Object from Model Parameters

            Args:
                model_parameters_path (str): Filename of the model parameters CSV or XML that
                    describes the case to be analysed
        """
        self.verbose = verbose
        # Initialize the Params Object from Model Parameters
        self.case_dict = Params.initialize(model_parameters_path, verbose)  # unvalidated case instances
        self.results = Result.initialize(Params.results_inputs, Params.case_definitions)
        if verbose:
            self.visualization = Visualization(Params)
            self.visualization.class_summary()

    def solve(self):
        """ Run storageVET

        Returns: the Results class

        """
        starts = time.time()
        for key, value in self.case_dict.items():
            run = Scenario(value)
            run.set_up_poi_and_service_aggregator()
            run.initialize_cba()
            run.fill_and_drop_extra_data()
            run.optimize_problem_loop()

            Result.add_instance(key, run)  # cost benefit analysis is in the Result class

        Result.sensitivity_summary()

        ends = time.time()
        TellUser.info("Full runtime: " + str(ends - starts))
        return Result
