"""
Result.py

"""

__author__ = 'Halley Nathwani, Thien Nyguen, Kunle Awojinrin'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'


import pandas as pd
import logging
import copy
import numpy as np
from pathlib import Path
import os
import Finances as Fin

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Result:
    """ This class serves as the later half of DER-VET's 'case builder'. It collects all optimization results, preforms
    any post optimization calculations, and saves those results to disk. If there are multiple

    """
    # these variables get read in upon importing this module
    instances = {}
    sensitivity_df = pd.DataFrame()
    sensitivity = False
    dir_abs_path = ''

    @classmethod
    def initialize(cls, results_params, df_analysis):
        """ Initialized the class with inputs that persist across all instances.

        If there are multiple runs, then set up comparison of each run.

        Args:
            results_params (Dict): user-defined inputs from the model parameter inputs
            df_analysis (DataFrame): this is a dataframe of possible sensitivity analysis instances

        Returns:

        """
        cls.instances = {}
        cls.dir_abs_path = results_params['dir_absolute_path']
        cls.csv_label = results_params['label']
        if cls.csv_label == 'nan':
            cls.csv_label = ''
        cls.sensitivity_df = df_analysis

        # data frame of all the sensitivity instances
        cls.sensitivity = (not cls.sensitivity_df.empty)
        if cls.sensitivity:
            # edit the column names of the sensitivity df to be human readable
            human_readable_names = []
            for i, col_name in enumerate(cls.sensitivity_df.columns):
                human_readable_names.append('[SP]' + col_name[0] + ' ' + col_name[1])
            # self.sens_df = pd.DataFrame()
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
        cls.template = cls(scenario)
        cls.instances.update({key: cls.template})
        cls.template.post_analysis()
        cls.template.calculate_cba()
        cls.template.save_as_csv(key, cls.sensitivity)

    def __init__(self, scenario, cba_module=Fin.Financial):
        """ Initialize a Result object, given a Scenario object with the following attributes.

            Args:
                scenario (Scenario.Scenario): scenario object after optimization has run to completion
                cba_module (Object): the module that is used to initialize the CBA
        """
        self.active_objects = scenario.active_objects
        self.customer_sided = scenario.customer_sided
        self.frequency = scenario.frequency
        self.dt = scenario.dt
        self.verbose_opt = scenario.verbose_opt
        self.n = scenario.n
        self.n_control = scenario.n_control
        self.mpc = scenario.mpc

        self.start_year = scenario.start_year
        self.end_year = scenario.end_year
        self.opt_years = scenario.opt_years
        self.incl_site_load = scenario.incl_site_load
        self.incl_binary = scenario.incl_binary
        self.incl_slack = scenario.incl_slack
        self.power_growth_rates = scenario.growth_rates
        self.technologies = scenario.technologies
        self.services = scenario.services
        self.predispatch_services = scenario.predispatch_services
        self.financials = cba_module(scenario.finance_inputs)
        self.verbose = scenario.verbose
        self.objective_values = scenario.objective_values

        # outputted DataFrames
        self.dispatch_map = pd.DataFrame()
        self.peak_day_load = pd.DataFrame()
        self.results = pd.DataFrame(index=scenario.optimization_subsets.index)
        self.energyp_map = pd.DataFrame()
        self.analysis_profit = pd.DataFrame()
        self.adv_monthly_bill = pd.DataFrame()
        self.sim_monthly_bill = pd.DataFrame()
        self.monthly_data = pd.DataFrame()
        self.deferral_dataframe = None
        self.technology_summary = None
        self.demand_charges = None

    def post_analysis(self):
        """ Wrapper for Post Optimization Analysis. Depending on what the user wants and what services were being
        provided, analysis on the optimization solutions are completed here.

        TODO: [multi-tech] a lot of this logic will have to change with multiple technologies
        """

        print("Performing Post Optimization Analysis...") if self.verbose else None

        # add other helpful information to a RESULTS DATAFRAME
        self.results.loc[:, 'Total Load (kW)'] = 0
        self.results.loc[:, 'Total Generation (kW)'] = 0
        # collect all storage power to handle multiple storage technologies, similar to total generation
        self.results.loc[:, 'Total Storage Power (kW)'] = 0
        self.results.loc[:, 'Aggregated State of Energy (kWh)'] = 0

        # collect results from technologies
        tech_type = []
        tech_name = []

        #  output to timeseries_results.csv
        for name, tech in self.technologies.items():
            if 'Deferral' not in self.predispatch_services.keys() or len(self.services.keys()):
                # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
                # then we did not optimize (skipped the optimization loop) NOTE - INVERSE OF CONDITIONAL ON LINE 547 in STORAGEVET\SCENARIO.PY
                report_df = tech.timeseries_report()
                self.results = pd.concat([report_df, self.results], axis=1)

                if name == 'PV':
                    self.results.loc[:, 'Total Generation (kW)'] += self.results['PV Generation (kW)']
                if name == 'ICE':
                    self.results.loc[:, 'Total Generation (kW)'] += self.results['ICE Generation (kW)']
                if name == 'CHP':
                    self.results.loc[:, 'Total Generation (kW)'] += self.results[tech.name + ' CHP Generation (kW)']
                if name == 'Storage':
                    self.results.loc[:, 'Total Storage Power (kW)'] += self.results[tech.name + ' Power (kW)']
                    self.results.loc[:, 'Aggregated State of Energy (kWh)'] += self.results[tech.name + ' State of Energy (kWh)']
                if name == 'Load':
                    self.results.loc[:, 'Total Load (kW)'] += self.results['Load (kW)']
            tech_name.append(tech.name)
            tech_type.append(tech.type)

        self.technology_summary = pd.DataFrame({'Type': tech_type}, index=pd.Index(tech_name, name='Name'))

        # collect results from each value stream
        for service in self.services.values():
            report_df = service.timeseries_report()
            self.results = pd.concat([self.results, report_df], axis=1)
            report = service.monthly_report()
            self.monthly_data = pd.concat([self.monthly_data, report], axis=1, sort=False)

        for pre_dispatch in self.predispatch_services.values():
            report_df = pre_dispatch.timeseries_report()
            self.results = pd.concat([self.results, report_df], axis=1)
            report = pre_dispatch.monthly_report()
            self.monthly_data = pd.concat([self.monthly_data, report], axis=1, sort=False)

        # assumes the original net load only does not contain the Storage system
        # self.results.loc[:, 'Original Net Load (kW)'] = self.results['Total Load (kW)']
        self.results.loc[:, 'Net Load (kW)'] = self.results['Total Load (kW)'] - self.results['Total Generation (kW)'] - self.results['Total Storage Power (kW)']

        if 'DCM' in self.active_objects['value streams']:
            self.demand_charges = self.services['DCM'].tariff

        if 'retailTimeShift' in self.active_objects['value streams']:
            energy_price = self.results.loc[:, 'Energy Price ($/kWh)'].to_frame()
            energy_price.loc[:, 'date'] = self.results.index.date
            energy_price.loc[:, 'hour'] = (self.results.index + pd.Timedelta('1s')).hour + 1  # hour ending
            energy_price = energy_price.reset_index(drop=True)
            self.energyp_map = energy_price.pivot_table(values='Energy Price ($/kWh)', index='hour', columns='date')

        if "DA" in self.services.keys():
            energy_price = self.results.loc[:, 'DA Price Signal ($/kWh)'].to_frame()
            energy_price.loc[:, 'date'] = self.results.index.date
            energy_price.loc[:, 'hour'] = (self.results.index + pd.Timedelta('1s')).hour + 1  # hour ending
            energy_price = energy_price.reset_index(drop=True)
            self.energyp_map = energy_price.pivot_table(values='DA Price Signal ($/kWh)', index='hour', columns='date')

        if 'Deferral' in self.active_objects['value streams']:
            self.deferral_dataframe = self.predispatch_services['Deferral'].deferral_df
            # these try to capture the import power to the site pre and post storage
            if self.deferral_dataframe is None:
                self.results.loc[:, 'Pre-storage Net Power (kW)'] = self.results['Total Load (kW)'] - self.results['Total Generation (kW)']
                self.results.loc[:, 'Pre-storage Net Power (kW)'] += self.results['Deferral Load (kW)']
                self.results.loc[:, 'Post-storage Net Power (kW)'] = self.results['Pre-storage Net Power (kW)']
                for name, tech in self.technologies.items():
                    self.results.loc[:, 'Post-storage Net Power (kW)'] = self.results['Post-storage Net Power (kW)'] - self.results[tech.name + ' Power (kW)']

        # create DISPATCH MAP dictionary to handle multiple storage technologies
        if self.deferral_dataframe is None:
            self.dispatch_map = {}
            for name, tech in self.technologies.items():
                if name == 'Storage':
                    dispatch = self.results.loc[:, tech.name + ' Power (kW)'].to_frame()
                    dispatch.loc[:, 'date'] = self.results.index.date
                    dispatch.loc[:, 'hour'] = (self.results.index + pd.Timedelta('1s')).hour + 1  # hour ending
                    dispatch = dispatch.reset_index(drop=True)

                    self.dispatch_map = dispatch.pivot_table(values=tech.name + ' Power (kW)', index='hour', columns='date')

        u_logger.debug("Finished post optimization analysis")

    def calculate_cba(self):
        """ Calls all finacial methods that will result in a series of dataframes to describe the cost benefit analysis for the
        case in question.

        """
        value_streams = {**self.services, **self.predispatch_services}
        if 'Deferral' not in self.predispatch_services.keys() or len(self.services.keys()):
            # if Deferral is on, and there is no energy market specified for energy settlement (or other market services)
            # then we did not optimize (skipped the optimization loop) NOTE - INVERSE OF CONDITIONAL ON LINE 547 in STORAGEVET\SCENARIO.PY
            self.financials.preform_cost_benefit_analysis(self.technologies, value_streams, self.results)

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
            savepath = self.dir_abs_path + "\\" + str(instance_key)
        else:
            savepath = self.dir_abs_path
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.results.sort_index(axis=1, inplace=True)  # sorts by column name alphabetically
        self.results.to_csv(path_or_buf=Path(savepath, 'timeseries_results' + self.csv_label + '.csv'))
        if self.customer_sided:
            self.financials.billing_period_bill.to_csv(path_or_buf=Path(savepath, 'adv_monthly_bill' + self.csv_label + '.csv'))
            self.financials.monthly_bill.to_csv(path_or_buf=Path(savepath, 'simple_monthly_bill' + self.csv_label + '.csv'))
            if 'DCM' in self.active_objects['value streams']:
                self.demand_charges.to_csv(path_or_buf=Path(savepath, 'demand_charges' + self.csv_label + '.csv'))

        self.peak_day_load.to_csv(path_or_buf=Path(savepath, 'peak_day_load' + self.csv_label + '.csv'))
        self.dispatch_map.to_csv(path_or_buf=Path(savepath, 'dispatch_map' + self.csv_label + '.csv'))

        if 'Deferral' in self.predispatch_services.keys():
            self.deferral_dataframe.to_csv(path_or_buf=Path(savepath, 'deferral_results' + self.csv_label + '.csv'))

        if 'DA' in self.services.keys() or 'retailTimeShift' in self.services.keys():
            self.energyp_map.to_csv(path_or_buf=Path(savepath, 'energyp_map' + self.csv_label + '.csv'))
        self.technology_summary.to_csv(path_or_buf=Path(savepath, 'technology_summary' + self.csv_label + '.csv'))

        # add other services that have monthly data here, if we want to save its monthly financial data report
        self.monthly_data.to_csv(path_or_buf=Path(savepath, 'monthly_data' + self.csv_label + '.csv'))

        ###############################
        # PRINT FINALCIAL/CBA RESULTS #
        ###############################
        self.financials.pro_forma.to_csv(path_or_buf=Path(savepath, 'pro_forma' + self.csv_label + '.csv'))
        self.financials.npv.to_csv(path_or_buf=Path(savepath, 'npv' + self.csv_label + '.csv'))
        self.financials.cost_benefit.to_csv(path_or_buf=Path(savepath, 'cost_benefit' + self.csv_label + '.csv'))
        self.financials.payback.to_csv(path_or_buf=Path(savepath, 'payback' + self.csv_label + '.csv'))

        if self.verbose:
            self.objective_values.to_csv(path_or_buf=Path(savepath, 'objective_values' + self.csv_label + '.csv'))
        print('Results have been saved to: ' + savepath)

    @classmethod
    def sensitivity_summary(cls):
        """ Loop through all the Result instances to build the dataframe capturing the important financial results
        and unique sensitivity input parameters for all instances.
            Then save the dataframe to a csv file.

        """
        if cls.sensitivity:
            for key, results_object in cls.instances.items():
                if not key:
                    for npv_col in results_object.financials.npv.columns:
                        cls.sensitivity_df.loc[:, npv_col] = 0
                this_npv = results_object.financials.npv.reset_index(drop=True, inplace=False)
                this_npv.index = pd.RangeIndex(start=key, stop=key + 1, step=1)
                cls.sensitivity_df.update(this_npv)

            cls.sensitivity_df.to_csv(path_or_buf=Path(cls.dir_abs_path, 'sensitivity_summary.csv'))

    @classmethod
    def proforma_df(cls, instance=0):
        """ Return the financial pro_forma for a specific instance

        """
        return cls.instances[instance].financials.pro_forma
