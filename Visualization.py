__author__ = 'Thien Nguyen and Halley Nathwani'
__copyright__ = 'Copyright 2018. Electric Power Research Institute (EPRI). All Rights Reserved.'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani',
               'Micah Botkin-Levy', "Thien Nguyen", 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.1.1'

import xml.etree.ElementTree as et
import pandas as pd
import itertools
import logging
import copy
import numpy as np
import os.path
from pathlib import Path
from prettytable import PrettyTable
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Visualization:

    def __init__(self, params_class):
        self.params_class = params_class

    def verify_tariff(self):
        """
            TODO: Future implementation: - TN
            1) visualize the billing periods/demand rates of the user tariff data for overlapping periods (if any)
            2) edit the numbers on the front end so they could possibly be saved and integrated to the back-end tariff
            dataset framework for testing purposes
        """

        """
            Function to summarize the other non time-series data in a tabular format: customer tariff

            Return: the summary table
        """
        customer_tariff_dict = self.params_class.datasets["customer_tariff"]
        for ct in customer_tariff_dict:
            u_logger.info("Building table for Customer Tariff Data: %s", ct)
            data = pd.read_csv(ct, sep=',', header=None)
            labels = data.iloc[0, :].values
            table = PrettyTable()
            for column in range(len(labels)):
                li = data.loc[1:len(data.index), column].values
                table.add_column(labels[column], li)
            u_logger.info("Customer Tariff table: \n" + str(table))
            u_logger.info("Finished making table for Customer Tariff...")

        return table

    def monthly_data_summary(self):
        """
            Function to summarize the other non time-series data in a tabular format: monthly data

            Return: the summary table
        """
        monthly_data_dict = self.params_class.datasets["monthly_data"]
        for md in monthly_data_dict:
            u_logger.info("Building plots for Monthly Data: %s", md)
            data = pd.read_csv(md, sep=',', header=None)
            labels = data.iloc[0, :].values
            table = PrettyTable()
            for column in range(len(labels)):
                li = data.loc[1:len(data.index), column].values
                table.add_column(labels[column], li)
            u_logger.info("Monthly Data table: \n" + str(table))
            u_logger.info("Finished making table for Monthly Data...")

        return table

    @classmethod
    def battery_cycle_life_summary(self):
        """
            Function to summarize the other non time-series data in a tabular format: battery cycle life

            Return: the summary table
        """
        cycle_life_dict = self.params_class.datasets["cycle_life"]
        for cl in cycle_life_dict:
            u_logger.info("Building table for Battery Cycle Life Data: %s", cl)
            data = pd.read_csv(cl, sep=',', header=None)
            labels = data.iloc[0, :].values
            table = PrettyTable()
            for column in range(len(labels)):
                li = data.loc[1:len(data.index), column].values
                table.add_column(labels[column], li)
            u_logger.info("Battery Cycle Life table: \n" + str(table))
            u_logger.info("Finished making table for Battery Cycle Life...")

        return table

    def instance_summary(self):
        """
        Prints a summary of input data for a specific scenario/instance

        Returns: 0 when successfully complete
        """
        u_logger.info("Summary for an instance of sensitivity:")
        active_components = self.fill_active()
        for key, value in active_components.items():
            # currently just printing in the logging unless data sensitivity is involved more in the future
            u_logger.info("CATEGORY: %s", key)
            for v in value:
                element = getattr(self.params_class.template, v)
                c = copy.copy(element)
                c.pop('cycle_life', None)
                c.pop('monthly_data', None)
                c.pop('customer_tariff', None)
                c.pop('time_series', None)
                u_logger.info("Element %s: %s", v, str(c))
        u_logger.info("Finished making instance summary for sensitivity.")

        return 0

    @classmethod
    def fill_active(cls):
        """ if given Tag is active it adds it to the class variable 'active_components'.
            checks to find the Tag category (i.e. service and pre-dispatch) through the schema.

        Returns each active component organized into a dictionary
        """
        root = cls.xmlTree.getroot()
        active_components = {"pre-dispatch": list(), "service": list(), "generator": list(), "load": list(),
                             "storage": list(), "der": list(), "scenario": list(),
                             "finance": list(), "command": list()}

        for tag in list(root):
            # check whether the tag is active.
            if tag.get('active')[0].lower() == "y" or tag.get('active')[0] == "1":

                tag_type = tag.get('type')

                if tag_type == 'pre-dispatch':
                    active_components['pre-dispatch'].append(tag.tag)
                elif tag_type == 'service':
                    active_components["service"].append(tag.tag)
                elif tag_type == 'generator':
                    active_components["generator"].append(tag.tag)
                elif tag_type == 'load':
                    active_components["load"].append(tag.tag)
                elif tag_type == 'storage':
                    active_components["storage"].append(tag.tag)
                elif tag_type == 'der':
                    active_components["der"].append(tag.tag)
                elif tag_type == 'scenario':
                    active_components["scenario"].append(tag.tag)
                elif tag_type == 'finance':
                    active_components["finance"].append(tag.tag)
        return active_components

    def series_summary(self):
        """ Function to plot all the time series data provided in each time series input file (if more than 1 file)

        Return: the figure of all the time series plots per input file name

        Notes: currently only return the figure of the last time series input file (if more than 1 file)

        """
        time_series_dict = self.params_class.datasets["time_series"]
        for ts in time_series_dict:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, squeeze=False, dpi=80, facecolor='w', edgecolor='k')
            fig.canvas.set_window_title('EPRI')
            # plt.subplots_adjust(right=0.70)
            # plt.tight_layout(rect=[0.04, 0.1, 0.85, 0.9])
            ts_name = ts
            dates_col = ['Datetime (he)']

            # select read function based on file type
            if ".csv" in ts_name:
                ts = pd.read_csv(ts, index_col=dates_col, parse_dates=dates_col, date_parser=lambda t: pd.datetime.strptime(t, "%m/%d/%Y %H:%M"))
            else:
                ts = pd.read_excel(ts, index_col=dates_col[0], parse_dates=dates_col, infer_datetime_format=True)

            # the plot is not optimized for time series data longer than 1 year and with timestep dt < 1 hour
            # TODO: what about a leap year? (len = 8784, when dt is 1)
            if len(ts) > 8761:
                return 0
            else:
                labels = list(ts.columns.values)
                priceindex = []
                loadindex = []
                priceindex2 = []
                price = []
                loads = []
                price2 = []

                # loading time series data to corresponding lists above
                # as well as their column index from the time series input file
                for column in range(len(labels)):
                    # these have unit as $/kWh
                    if "Price ($/kWh)" in labels[column]:
                        price2.append(ts.iloc[1:, column].values)
                        priceindex2.append(column)
                    # these have unit as $/kW
                    elif "Price ($/kW)" in labels[column]:
                        price.append(ts.iloc[1:, column].values)
                        priceindex.append(column)
                    # these have unit as kW
                    elif "Load (kW)" in labels[column]:
                        loads.append(ts.iloc[1:, column].values)
                        loadindex.append(column)
                    # temporarily skip all other time series data that don't have the above units
                    else:
                        continue

                axes[0, 0].title.set_text('Time Series Data for ' + ts_name)
                # adjust setting for each subplot
                for x in range(3):
                    box = axes[x, 0].get_position()
                    axes[x, 0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    axes[x, 0].set_autoscale_on(True)
                    axes[x, 0].autoscale_view(True, True, True)

                # plotting if these time series data is available
                if priceindex2:
                    ts.plot(y=priceindex2, linewidth='0.5', ax=axes[0, 0])
                    axes[0, 0].legend([labels[i] for i in priceindex2])
                    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fancybox=True, shadow=True,
                                      fontsize='small', borderaxespad=0.)
                    axes[0, 0].autoscale(enable=True, axis='y', tight=True)
                    # axes[0, 0].yaxis.set_ticks(np.arange(0, 1, 0.1))
                    axes[0, 0].yaxis.set_label_coords(-0.05, 1.1)
                    axes[0, 0].set_ylabel('Prices ($/kWh)', rotation=0)
                if priceindex:
                    ts.plot(y=priceindex, linewidth='0.5', ax=axes[1, 0])
                    axes[1, 0].legend([labels[i] for i in priceindex])
                    axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fancybox=True, shadow=True,
                                      fontsize='small', borderaxespad=0.)
                    axes[1, 0].autoscale(enable=True, axis='y', tight=True)
                    # axes[1, 0].yaxis.set_ticks(np.arange(0, 2, 0.2))
                    axes[1, 0].yaxis.set_label_coords(-0.05, 1.05)
                    axes[1, 0].set_ylabel("Prices ($/kW)", rotation=0)
                if loadindex:
                    ts.plot(y=loadindex, linewidth='0.5', ax=axes[2, 0])
                    axes[2, 0].legend([labels[i] for i in loadindex])
                    axes[2, 0].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, fancybox=True, shadow=True,
                                      fontsize='small', borderaxespad=0.)
                    axes[2, 0].autoscale(enable=True, axis='y', tight=True)
                    # axes[2, 0].yaxis.set_ticks(np.arange(0, 20001, 3000))
                    axes[2, 0].yaxis.set_label_coords(-0.05, 1.0)
                    axes[2, 0].set_ylabel("Loads (kW)", rotation=0)

            fig.autofmt_xdate()

            # uncomment this if you want to plot the time series data without using GUI - TN
            # plt.show()
            # plt.close(fig)

        return fig

    def class_summary(self):
        """
            Function to summarize the Model_Parameters_Template for previsualization

            Return: the summary table
        """
        tree = self.params_class.xmlTree
        treeRoot = tree.getroot()
        schema = self.params_class.schema_tree

        u_logger.info("Printing summary table for class Params")
        table = PrettyTable()
        table.field_names = ["Category", "Tag", "Active?", "Key", "Analysis?", "Value", "Value Type", "Sensitivity"]
        schema_root = schema.getroot()
        for tag in treeRoot:
            schemaType = self.search_schema_type(schema_root, tag.tag)
            activeness = tag.attrib.get('active')
            if len(tag) > 0:
                for key in tag:
                    table.add_row([schemaType, tag.tag, activeness, key.tag, key.attrib.get('analysis'),
                                   key.find('Value').text, key.find('Type').text, key.find('Sensitivity_Parameters').text])
            else:
                table.add_row([schemaType, tag.tag, activeness, '', '', '', '', ''])

        u_logger.info('Params class summary: \n' + str(table))

        print(table)

        return table

    @staticmethod
    def search_schema_type(root, component_name):
        """ Looks in the schema XML for the type of the component. Used to print the class summary for previsualization.
            Ex: storage-type, storage-technology, pre-dispatch, services, generator, finance, or scenario
        Args:
            root (collection.Iterable): the root of the XML input tree
            component_name (str): name of the attribute to be searched and determined for the type

        Returns: the type of the attribute, if found. otherwise it returns "other"

        """
        for child in root:
            attributes = child.attrib
            if child.tag == component_name:
                if attributes.get('type') is None:
                    return "other"
                else:
                    return attributes.get('type')

    def sens_summary(self):
        """
             Function to summarize the data frame for Sensitivity Analysis from Params Class
             Including the component names, its data variables, and all the possible data combinations of Sensitivity

             Return: the summary table
         """

        df = self.params_class.case_definitions
        u_logger.info("Printing sensitivity summary table for class Params")

        if len(df) == 0:
            return 0
        else:
            headers = ["Scenario"]
            for v in df.columns.values:
                headers.append(v)
            table2 = PrettyTable()
            table2.field_names = headers

            for index, row in df.iterrows():
                entry = [index]
                for col in row:
                    entry.append(str(col))
                table2.add_row(entry)

            u_logger.info("Sensitivity summary: \n" + str(table2))
            return table2