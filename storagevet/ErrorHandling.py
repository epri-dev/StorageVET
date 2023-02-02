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
This file contains all error classes. It also initialized and writes to the log
files to give the user feedback on what went wrong.
"""

import os
import logging
from pathlib import Path


class SolverError(Exception):
    pass


class SolverInfeasibleError(Exception):
    pass


class SolverUnboundedError(Exception):
    pass


class SystemRequirementsError(Exception):
    pass


class ModelParameterError(Exception):
    pass


class TimeseriesDataError(Exception):
    pass


class TimeseriesMissingError(Exception):
    pass


class MonthlyDataError(Exception):
    pass


class TariffError(Exception):
    pass


class ParameterError(ValueError):
    pass


class FilenameError(ValueError):
    pass


class TellUser:
    @classmethod
    def create_log(cls, logs_path, verbose):
        try:
            os.makedirs(logs_path)
        except OSError:
            print("Creation of the logs_path directory %s failed. Possibly already created." % logs_path) if verbose else None
        else:
            print("Successfully created the logs_path directory %s " % logs_path) if verbose else None
        log_filename = logs_path / 'dervet_log.log'
        handler = logging.FileHandler(log_filename, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        cls.logger = logging.getLogger('Error')
        cls.logger.setLevel(logging.DEBUG)
        cls.logger.addHandler(handler)
        if verbose:
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # add formatter to ch
            ch.setFormatter(formatter)
            # add ch to logger
            cls.logger.addHandler(ch)
        cls.logger.info('Started logging...')

    @classmethod
    def close_log(cls):
        for i in list(cls.logger.handlers):
            print(i)
            cls.logger.removeHandler(i)
            i.flush()
            i.close()

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)
