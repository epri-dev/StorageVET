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
Result.py

"""
import pandas as pd
from storagevet.ErrorHandling import *


class Result:
    """ This class serves as the later half of DER-VET's 'case builder'. It collects all optimization results, preforms
    any post optimization calculations, and saves those results to disk. If there are multiple

    """
    # these variables get read in upon importing this module
    instances = None
    sensitivity_df = None
    sensitivity = False
    dir_abs_path = None

    @classmethod
    def initialize(cls, results_params, case_definitions):
        """ Initialized the class with inputs that persist across all instances.

        If there are multiple runs, then set up comparison of each run.

        Args:
            results_params (Dict): user-defined inputs from the model parameter inputs
            case_definitions (DataFrame): this is a dataframe of possible sensitivity analysis instances

        Returns:

        """
        cls.instances = {}
        cls.dir_abs_path = Path(results_params['dir_absolute_path'])
        cls.csv_label = results_params.get('label', '')
        if cls.csv_label == 'nan':
            cls.csv_label = ''
        cls.sensitivity_df = case_definitions

        # data frame of all the sensitivity instances
        cls.sensitivity = (not cls.sensitivity_df.empty)
        if cls.sensitivity:
            # edit the column names of the sensitivity df to be human readable
            human_readable_names = [f"[SP] {col_name[0]} {col_name[0]}" for col_name in cls.sensitivity_df.columns]
            # TODO: use scehma.xml to get unit of tag-key combo and add to HUMAN_READABLE_NAMES
            cls.sensitivity_df.columns = human_readable_names
            cls.sensitivity_df.index.name = 'Case Number'

    @classmethod
    def add_instance(cls, key, scenario):
        """

        Args:
            key (int): the key that corresponds to the value this instance corresponds to within the df_analysis
                dataFrame from the Params class.
            scenario (Scenario.Scenario): scenario object after optimization has run to completion

        """
        # initialize an instance of Results
        template = cls(scenario)
        # save it in our dictionary of instance (so we can keep track of all the Results we have)
        cls.instances.update({key: template})
        # preform post facto calculations and CBA
        template.collect_results()
        template.create_drill_down_dfs()
        template.calculate_cba()
        # save dataframes as CSVs
        template.save_as_csv(key, cls.sensitivity)

    def __init__(self, scenario):
        """ Initialize a Result object, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
        """
        self.frequency = scenario.frequency
        self.dt = scenario.dt
        self.verbose_opt = scenario.verbose_opt
        self.n = scenario.n
        self.n_control = scenario.n_control
        self.mpc = scenario.mpc
        self.start_year = scenario.start_year
        self.end_year = scenario.end_year
        self.opt_years = scenario.opt_years
        self.incl_binary = scenario.incl_binary
        self.incl_slack = scenario.incl_slack
        self.verbose = scenario.verbose
        self.poi = scenario.poi
        self.service_agg = scenario.service_agg
        self.objective_values = scenario.objective_values
        self.cost_benefit_analysis = scenario.cost_benefit_analysis
        self.opt_engine = scenario.opt_engine

        # initialize DataFrames that drill down dfs will be built off
        self.time_series_data = pd.DataFrame(index=scenario.optimization_levels.index)
        self.monthly_data = pd.DataFrame()
        self.technology_summary = pd.DataFrame()

        # initialize the dictionary that will hold all the drill down plots
        self.drill_down_dict = dict()

    def collect_results(self):
        """ Collects any optimization variable solutions or user inputs that will be used for drill down
        plots, as well as reported to the user. No matter what value stream or DER is being evaluated, these
        dataFrames should always be made and reported to the user

        Three attributes are edited in this method: TIME_SERIES_DATA, MONTHLY_DATA, TECHNOLOGY_SUMMARY
        """

        TellUser.debug("Performing Post Optimization Analysis...")

        report_df, monthly_report = self.poi.merge_reports(self.opt_engine,
                                                           self.time_series_data.index)
        self.time_series_data = pd.concat([self.time_series_data, report_df],
                                          axis=1)
        self.monthly_data = pd.concat([self.monthly_data, monthly_report],
                                      axis=1, sort=False)

        # collect results from each value stream
        ts_df, month_df = self.service_agg.merge_reports()
        self.time_series_data = pd.concat([self.time_series_data, ts_df], axis=1)

        self.monthly_data = pd.concat([self.monthly_data, month_df], axis=1, sort=False)

        self.technology_summary = self.poi.technology_summary()

    def create_drill_down_dfs(self):
        """ Tells ServiceAggregator and POI to build drill down reports. These are reports
        that are service or technology specific.

        Returns: Dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with

        """
        if self.opt_engine:
            self.drill_down_dict.update(self.poi.drill_down_dfs(monthly_data=self.monthly_data, time_series_data=self.time_series_data,
                                                                technology_summary=self.technology_summary))
        self.drill_down_dict.update(self.service_agg.drill_down_dfs(monthly_data=self.monthly_data, time_series_data=self.time_series_data,
                                                                    technology_summary=self.technology_summary))
        TellUser.info("Finished post optimization analysis")

    def calculate_cba(self):
        """ Calls all finacial methods that will result in a series of dataframes to describe the cost benefit analysis for the
        case in question.

        """
        self.cost_benefit_analysis.calculate(self.poi.der_list, self.service_agg.value_streams, self.time_series_data, self.opt_years)

    def save_as_csv(self, instance_key, sensitivity=False):
        """ Save useful DataFrames to disk in csv files in the user specified path for analysis.

        Args:
            instance_key (int): string of the instance value that corresponds to the Params instance that was used for
                this simulation.
            sensitivity (boolean): logic if sensitivity analysis is active. If yes, save_path should create additional
                subdirectory

        Prints where the results have been saved when completed.
        """
        if sensitivity:
            savepath = self.dir_abs_path / str(instance_key)
        else:
            savepath = self.dir_abs_path
        if not savepath.exists():
            os.makedirs(savepath)

        suffix = f"{self.csv_label}.csv"

        # time series
        self.time_series_data.index.rename('Start Datetime (hb)', inplace=True)
        self.time_series_data.sort_index(axis=1, inplace=True)  # sorts by column name alphabetically
        self.time_series_data.to_csv(path_or_buf=Path(savepath, f'timeseries_results{suffix}'))
        # monthly data
        self.monthly_data.to_csv(path_or_buf=Path(savepath, f'monthly_data{suffix}'))
        # technology summary
        self.technology_summary.to_csv(path_or_buf=Path(savepath, f'technology_summary{suffix}'))

        # save the drill down dfs  NOTE lists are faster to iterate through -- HN
        for file_name, df in self.drill_down_dict.items():
            df.to_csv(path_or_buf=Path(savepath, f"{file_name}{suffix}"))
        # PRINT FINALCIAL/CBA RESULTS
        finacials_dfs = self.cost_benefit_analysis.report_dictionary()
        for file_name, df in finacials_dfs.items():
            df.to_csv(path_or_buf=Path(savepath, f"{file_name}{suffix}"))

        if self.verbose:
            self.objective_values.to_csv(path_or_buf=Path(savepath, f'objective_values{suffix}'))
        TellUser.info(f'Results have been saved to: {savepath}')

    @classmethod
    def sensitivity_summary(cls):
        """ Loop through all the Result instances to build the dataframe capturing the important financial results
        and unique sensitivity input parameters for all instances.
            Then save the dataframe to a csv file.

        """
        if cls.sensitivity:
            for key, results_object in cls.instances.items():
                if not key:
                    for npv_col in results_object.cost_benefit_analysis.npv.columns:
                        cls.sensitivity_df.loc[:, npv_col] = 0
                this_npv = results_object.cost_benefit_analysis.npv.reset_index(drop=True, inplace=False)
                if this_npv.empty:
                    # then no optimization ran, so there is no NPV
                    continue
                this_npv.index = pd.RangeIndex(start=key, stop=key + 1, step=1)
                cls.sensitivity_df.update(this_npv)
            cls.sensitivity_df.to_csv(path_or_buf=Path(cls.dir_abs_path, 'sensitivity_summary.csv'))

    @classmethod
    def proforma_df(cls, instance=0):
        """ Return the financial pro_forma for a specific instance

        """
        return cls.instances[instance].cost_benefit_analysis.pro_forma
