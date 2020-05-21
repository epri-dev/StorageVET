"""
Params.py

"""

__author__ = 'Yekta Yazar, Evan Giarta, Thien Nguygen, Halley Nathwani'
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
from datetime import datetime
from matplotlib.font_manager import FontProperties
from Finances import Financial
import Library as Lib

e_logger = logging.getLogger('Error')
u_logger = logging.getLogger('User')


class Params:
    """
        Class attributes are made up of services, technology, and any other needed inputs. The attributes are filled
        by converting the xml file in a python object.

        Notes:
             Might need to modify the summary functions for pre-visualization every time the Params class is changed
             in terms of the naming of the class variables, the active components and their dictionary structure - TN
    """
    # set schema loction based on the location of this file (this should override the global value within Params.py
    schema_location = os.path.abspath(__file__)[:-len('Params.py')] + "Schema.xml"

    # initialize class variables
    schema_tree = None
    xmlTree = None

    active_tag_key = set()
    sensitivity = {"attributes": dict(), "coupled": list()}  # contains all sensitivity variables as keys and their values as lists
    case_definitions = pd.DataFrame()  # Each row specifies the value of each attribute

    instances = dict()
    template = None
    referenced_data = {
        "time_series": dict(),
        "monthly_data": dict(),
        "customer_tariff": dict(),
        "cycle_life": dict(),
        "yearly_data": dict()
    }  # for a scenario in the sensitivity analysis
    # holds the data of all the time series usd in sensitivity analysis

    results_inputs = {
        'label': '',
        'dir_absolute_path': './Results/' + '_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y'),
        'errors_log_path': './Results/' + '_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y')
    }

    input_error_raised = False

    @classmethod
    def initialize(cls, filename, verbose):
        """ This function 1) converts CSV into XML; 2) read in the XML; 3) convert indirect data references into direct data;
        4) create instances from Sensitivity Analysis and Coupling values

            Args:
                filename (string): filename of XML or CSV model parameter
                verbose (bool): whether or not to print to console for more feedback

            Returns dictionary of instances of Params, each key is a number
        """
        # 1) INITIALIZE ALL MUTABLE CLASS VARIABLES
        cls.sensitivity = {"attributes": dict(), "coupled": list()}  # contains all sensitivity variables as keys and their values as lists
        cls.case_definitions = pd.DataFrame()  # Each row specifies the value of each attribute for sensitivity analysis

        cls.instances = dict()
        cls.input_error_raised = False

        # holds the data of all the time series usd in sensitivity analysis
        cls.referenced_data = {
            "time_series": dict(),
            "monthly_data": dict(),
            "customer_tariff": dict(),
            "cycle_life": dict(),
            "yearly_data": dict()}

        cls.results_inputs = {'label': '',
                              'dir_absolute_path': './Results/' + '_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y'),
                              'errors_log_path': './Results/' + '_' + datetime.now().strftime('%H_%M_%S_%m_%d_%Y')}

        # 2) CONVERT CSV INTO XML
        if filename.endswith(".csv"):
            filename = cls.csv_to_xml(filename, verbose)

        # 3) LOAD DIRECT DATA FROM XML
        print('Initializing the XML tree with the provided file...') if verbose else None
        cls.xmlTree = et.parse(filename)
        cls.schema_tree = et.parse(cls.schema_location)

        # read Results tag and set as class attribute
        result_dic = cls.read_and_validate('Results')  # note that sensitivity analysis does not work on the 'Results' tag. this is by design.
        if result_dic:
            cls.results_inputs = result_dic
        cls.setup_logger(verbose)  # set up logger before reading in data (so we can report errors back)

        # _init_ the Params class
        cls.template = cls()

        # report back any warning
        if cls.input_error_raised:
            raise AssertionError("The model parameter has some errors associated to it. Please fix and rerun.")

        # turn referenced data into direct data and preprocess some
        cls.read_referenced_data()

        # 4) SET UP SENSITIVITY ANALYSIS AND COUPLING W/ CASE BUILDER
        print('Updating data for Sensitivity Analysis...') if verbose else None

        # build each of the cases that need to be run based on the model parameter inputed
        cls.build_case_definitions()
        cls.case_builder()

        return cls.instances

    @staticmethod
    def csv_to_xml(csv_filename, verbose=False):
        """ converts csv to xml

        Args:
            csv_filename (string): name of csv file
            verbose (bool): whether or not to print to console for more feedback

        # TODO: create a 'write_xml' function for dervet to use as well --HN
        # TODO: use XML API (library) to write the xml document --HN

        Returns:
            xmlFile (string): name of xml file
        """
        # find .csv in the filename and replace with .xml
        xml_filename = csv_filename[:csv_filename.rfind('.')] + ".xml"

        # open csv to read into dataframe and blank xml file to write to
        csv_data = pd.read_csv(csv_filename)
        xml_data = open(xml_filename, 'w')

        # write the header of the xml file and specify columns to place in xml model parameters template
        xml_data.write('<?xml version="1.0" encoding="UTF-8"?>' + "\n")
        xml_data.write('\n<input>\n')
        xml_columns = set(csv_data.columns) - {'Tag', 'Key', 'Units', 'Allowed Values', 'Description', 'Active',
                                               'Sensitivity Analysis', 'Evaluation Value', 'Evaluation Active', 'Options/Notes'}
        # xmlColumns = ['Optimization Value', 'Type', 'Sensitivity Parameters', 'Coupled']

        # outer loop for each tag/object and active status, i.e. Scenario, Battery, DA, etc.
        for obj in csv_data.Tag.unique():
            mask = csv_data.Tag == obj
            xml_data.write('\n    <' + obj.strip() + ' active="' + csv_data[mask].Active.iloc[0].strip() + '">\n')
            # middle loop for each object's elements and is sensitivity is needed: max_ch_rated, ene_rated, price, etc.
            for ind, row in csv_data[mask].iterrows():
                if row['Key'] is pd.np.nan:
                    continue
                xml_data.write('        <' + str(row['Key'].strip()) + ' analysis="' + str(row['Sensitivity Analysis']) + '">\n')
                # inner loop for specifying elements Value, Type, Coupled, etc.
                for tag in xml_columns:
                    if tag == 'Optimization Value':
                        tag_col = 'Value'
                    else:
                        tag_col = tag
                    xml_data.write(
                        '            <' + str(tag_col).replace(' ', '_') + '>' + str(row[tag]) + '</' + str(tag_col).replace(' ', '_') + '>\n')
                xml_data.write('        </' + str(row['Key']) + '>\n')
            xml_data.write('    </' + obj + '>\n')
        xml_data.write('\n</input>')
        xml_data.close()

        return xml_filename

    @classmethod
    def setup_logger(cls, verbose):
        logs_path = cls.results_inputs['errors_log_path']
        try:
            os.makedirs(logs_path)
        except OSError:
            print("Creation of the logs_path directory %s failed. Possibly already created." % logs_path) if verbose else None
        else:
            print("Successfully created the logs_path directory %s " % logs_path) if verbose else None
        log_filename = logs_path + "\\errors_log.log"
        e_handler = logging.FileHandler(Path(log_filename), mode='w')
        e_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        e_logger = logging.getLogger('Error')
        e_logger.setLevel(logging.ERROR)
        e_logger.addHandler(e_handler)
        e_logger.error('Started logging...')
        log_filename2 = logs_path + "\\user_log.log"
        u_handler = logging.FileHandler(Path(log_filename2))
        u_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        u_logger = logging.getLogger('User')
        u_logger.setLevel(logging.INFO)
        u_logger.addHandler(u_handler)
        u_logger.info('Started logging...')
        fontP = FontProperties()
        fontP.set_size('small')

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.

        """
        self.PV = self.read_and_validate('PV')
        self.Battery = self.read_and_validate('Battery')
        self.CAES = self.read_and_validate('CAES')
        self.ICE = self.read_and_validate('ICE')

        self.Scenario = self.read_and_validate('Scenario')
        self.Finance = self.read_and_validate('Finance')

        # bulk value streams/ market services
        self.DA = self.read_and_validate('DA')
        self.FR = self.read_and_validate('FR')
        self.SR = self.read_and_validate('SR')
        self.NSR = self.read_and_validate('NSR')
        self.LF = self.read_and_validate('LF')

        # customer-sited value streams
        self.DCM = self.read_and_validate('DCM')
        self.retailTimeShift = self.read_and_validate('retailTimeShift')

        # constraint value streams
        self.Backup = self.read_and_validate('Backup')  # this is an empty dictionary
        self.Deferral = self.read_and_validate('Deferral')
        self.User = self.read_and_validate('User')

        # value streams under construction
        self.DR = self.read_and_validate('DR')
        self.RA = self.read_and_validate('RA')
        self.RT = self.read_and_validate('RT')
        self.Volt = self.read_and_validate('Volt')

        self.POI = None  # save attribute to be set after datasets are loaded and all instances are made
        self.Load = None  # save attribute to be set after datasets are loaded and all instances are made

    @classmethod
    def read_and_validate(cls, name):
        """ Read data from xml file (usually after conversion from model parameter CSV) and initializes senesitivity
        analysis. Makes note of which Key in what Tag has sensitivity analysis by assigning a tuple of the key
        and property name as the index to the sensitivity values given. This is stored in the attributes index of
        the sensitivity dictionary, which is a nested dictionary whose first index is a string and second index
        is a tuple

        Args:
            name (str): name of the root 'Tag' in the xml file

        Returns:
            A dictionary filled with needed given values of the properties of each service/technology
            if the service or technology is not implemented in this case, then None is returned.
        """
        tag = cls.xmlTree.find(name)
        schema_tag = cls.schema_tree.find(name)

        # this catches a misspelling in the Params 'name' compared to the xml trees spelling of 'name'
        if tag is None:
            return None
        dictionary = None
        # This statement checks if the first character is 'y' or '1', if true it creates a dictionary.
        if tag.get('active')[0].lower() == "y" or tag.get('active')[0] == "1":
            # Check if tag is in schema
            if schema_tag is None:
                cls.report_warning("missing tag", tag=name, raise_input_error=False)
                # warn user that the tag given is not in the schema
                return dictionary
            dictionary = {}
            # iterate through each tag required by the schema
            for schema_key in schema_tag:
                # Check if attribute is in the schema
                try:
                    key = tag.find(schema_key.tag)
                    value = key.find('Value').text
                except (KeyError, AttributeError):
                    cls.report_warning("missing key", tag=name, key=schema_key.tag)
                    continue
                # check to see if input is optional
                if value == 'nan' and schema_key.get('optional') != 'y' and schema_key.get('optional') != '1':
                    cls.report_warning("not allowed", tag=name, key=schema_key.tag, value=value, allowed_values="Something other than 'nan'")
                    continue
                # convert to correct data type
                intended_type = schema_key.get('type')
                # fills dictionary with the base case from the xml file (this will serve as the template case)
                base_values = cls.convert_data_type(value, intended_type)
                # validate the inputted value
                cls.checks_for_validate(base_values, schema_key, name)
                # save value to base case
                dictionary[schema_key.tag] = base_values
                # if yes, assign this to the class's sensitivity variable.
                if key.get('analysis')[0].lower() == "y" or key.get('analysis')[0] == "1":
                    tag_key = (tag.tag, key.tag)
                    sensitivity_values = cls.extract_data(key.find('Sensitivity_Parameters').text, intended_type)
                    # validate each value
                    for values in sensitivity_values:
                        cls.checks_for_validate(values, schema_key, name)
                    # save all values to build into cases later
                    cls.sensitivity['attributes'][tag_key] = sensitivity_values
                    # check if the property should be Coupled
                    if key.find('Coupled').text != 'None':
                        coupled_set = cls.parse_coupled(tag.tag, key)
                        cls.fetch_coupled(coupled_set, tag_key)
        return dictionary

    @staticmethod
    def convert_data_type(value, desired_type):
        """ coverts data to a given type. takes in a value and a type

            Args:
                value (str): some data needed to be converted
                desired_type (str): the desired type of the given value

            Returns:
                Attempts to convert the type of 'value' to 'desired_type'. Returns value if it can,
                otherwise it returns a tuple (None, value) if desired_type is not a known type to the function or the value
                cannot be converted to the desired type.
                report if there was an error converting the given data in the convert_data_type method

            Notes if None or NaN is given for the value of VALUE, then this function will error
        """

        if desired_type == 'string':
            return value
        elif desired_type == "float":
            try:
                float(value)
                return float(value)
            except ValueError:
                return None, value
        elif desired_type == 'tuple':
            try:
                return tuple(value)
            except ValueError:
                return None, value
        elif desired_type == 'list':
            try:
                return list(value)
            except ValueError:
                return None, value
        elif desired_type == 'list/int':
            try:
                return list(map(int, value.split()))
            except ValueError:
                return None, value
        elif desired_type == 'bool':
            try:
                return bool(int(value))
            except ValueError:
                return None, value
        elif desired_type == 'Timestamp':
            try:
                return pd.Timestamp(value)
            except ValueError:
                return None, value
        elif desired_type == 'Period':
            try:
                return pd.Period(value)
            except ValueError:
                return None, value
        elif desired_type == "int":
            try:
                return int(value)
            except ValueError:
                return None, value
        elif desired_type == 'string/int':
            try:
                return int(value)
            except ValueError:
                return value
        elif desired_type == 'list/string':
            try:
                return value.split()
            except ValueError:
                return None, value
        else:
            return None, value

    @classmethod
    def checks_for_validate(cls, value, schema_key, tag_name):
        """ Helper function to validate method. This runs the checks to validate the value of input data with
            its properties (the arguments) provided from Schema.xml

            Args:
                value: the value of the specific key being looked at, as read in from the xml
                schema_key (Element): schema key for the value being validated
                tag_name (string): the tag from which this property is organized in

            Returns: error as a tuple of information (tag, key, what caused the error, the type of error, what the error should be changed into)
                if no error, then return None

        """

        desired_type = schema_key.get('type')
        # check if there was a conversion error
        if type(value) == tuple:
            cls.report_warning('conversion', value=value[1], type=desired_type, key=schema_key.tag, tag=tag_name)
            return

        # check if the data belongs to a set of allowed values
        allowed_values = schema_key.get('allowed_values')
        if allowed_values is not None:
            allowed_values = {cls.convert_data_type(item, desired_type) for item in allowed_values.split('|')}  # make a list of the allowed values
            # if the data type is 'STRING', then make sure to convert everything to lower case and then compare
            if desired_type == 'string':
                value = value.lower()
                allowed_values = {check_value.lower() for check_value in allowed_values}
            if not len({value}.intersection(allowed_values)):
                # check whether to consider a numerical bound for the set of possible values
                if 'BOUND' not in allowed_values:
                    cls.report_warning('not allowed', value=value, allowed_values=allowed_values, key=schema_key.tag, tag=tag_name)
                    return

        # check if data is in valid range
        bound_error = False
        minimum = schema_key.get('min')
        if minimum is not None:
            minimum = cls.convert_data_type(minimum, desired_type)
            try:  # handle if minimum is None
                bound_error = bound_error or value < minimum
            except TypeError:
                pass
        maximum = schema_key.get('max')
        if maximum is not None:
            maximum = cls.convert_data_type(maximum, desired_type)
            try:  # handle if maximum is None
                bound_error = bound_error or value > maximum
            except TypeError:
                pass
        if bound_error:
            cls.report_warning('size', value=value, key=schema_key.tag, tag=tag_name, min=minimum, max=maximum)
        return

    @classmethod
    def report_warning(cls, warning_type, raise_input_error=True, **kwargs):
        """ Print a warning to the user log. Warnings are reported, but do not result in exiting.

        Args:
            warning_type (str): the classification of the warning to be reported to the user
            raise_input_error (bool): raise this warning as an error instead back to the user and stop running
                the program
            kwargs: elements about the warning that need to be reported to the user (like the tag and key that
                caused the error

        """
        if warning_type == "missing tag":
            e_logger.error(f"INPUT: {kwargs['tag']} cannot be found in schema, so no validation or data conversion will occur." +
                           f"Please define an XML tree for '{kwargs['tag']}' in the file at {cls.schema_location} if you want " +
                           "the inputs to be considered.")

        if warning_type == "unknown tag":
            e_logger.error(f"INPUT: {kwargs['tag']} is not used in this program and will be ignored." +
                           "Please check spelling and rerun if you would like the key's associated to be considered.")

        if warning_type == "missing key":
            e_logger.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} is not found in model parameter. Please include and rerun.")

        if warning_type == "conversion":
            e_logger.error(f"INPUT: error converting {kwargs['tag']}-{kwargs['key']}" +
                           f"...trying to convert '{kwargs['value']}' into {kwargs['type']}")

        if warning_type == 'size':
            e_logger.error(f"INPUT: {kwargs['tag']}-{kwargs['key']}  value ('{kwargs['value']}') is out of bounds. " +
                           f"Max: ({kwargs['max']}  Min: {kwargs['min']})")

        if warning_type == "not allowed":
            e_logger.error(f"INPUT: {kwargs['tag']}-{kwargs['key']}  value ('{kwargs['value']}') is not a valid input." +
                           f"The value should be one of the following: {kwargs['allowed_values']}")

        if warning_type == "cannot couple":
            e_logger.error(f"INPUT: Error occurred when trying to couple {kwargs['tag_key']} with {kwargs['schema_key']}." +
                           "Please fix and run again.")

        cls.input_error_raised = cls.input_error_raised or raise_input_error

    @classmethod
    def extract_data(cls, expression, data_type):
        """
            Function to extract out all the sensitivity or non-sensitivity values for each given expression,
            due to their string format. And then to convert these data values to the proper data types
            in prior to the return.

            Args:
                expression (str): the string of expression to be extracted
                data_type (str): the string of dataType

            Returns:
                result (list): the list with each element as a sensitivity value type-casted for analysis.
                               in the case of non-sensitivity this list is only one element in size.

        """
        result = []
        expression = expression.strip()
        if expression.startswith("[") and expression.endswith("]"):
            expression = expression[1:-1]
        sset = expression.split(',')

        # converts values into correct data type
        for s in sset:
            data = cls.convert_data_type(s.strip(), data_type)
            result.append(data)
        return result

    @staticmethod
    def parse_coupled(tag, key):
        """ Function to allow coupling across different types of Tags (Scenario, Finance, etc) by improving text parsing
            By filtering out the types (categories) and then mapping them to the respective property result.
            Create a list that contains coupled_sets of properties from different types or same type.
            Improve readability so that fetch_coupled can be done appropriately.

            Args:
                tag (str): string representation of tag of csv (top level of the XML)
                key (et.Element): attribute of the tag

            Returns: a set of tuples (Tag, key) that the KEY inputted is coupled with

        """
        # list of strings of the `Tag:Key` or `Key` listed in 'Coupled'
        coupled_with = [x.strip() for x in key.find('Coupled').text.split(',')]
        # check if each item in list reports its Tag (looking for strings that are of the form `Tag:Key`, else assume given TAG)
        coupled_with_tag = [tag if x.find(':') < 0 else x[:x.find(':')] for x in coupled_with]
        # strip the `Tag:` portion off any strings that follow the form `Tag:Key`
        coupled_with = [x if x.find(':') < 0 else x[x.find(':') + 1:] for x in coupled_with]
        # create set of tuples (tag, key) of tag-key values that the given key is coupled with
        tag_key_set = set(zip(coupled_with_tag, coupled_with))
        return tag_key_set

    @classmethod
    def fetch_coupled(cls, coupled_properties, tag_key):
        """ Function to determine and fetch in only the relevant and unique coupled sets
            to the class coupled sensitivity variable.
            It determines the coupled properties first and then it is compared to other existing coupled set to
            determine if there is any subset or intersection.
            Append it to the class coupled sensitivity variable under appropriate conditions.

            Args:
                coupled_properties (set): a set that contains coupled_sets of properties from different
                    types or same type.
                tag_key (tuple(str): tuple of the tag and key
        """
        # check to make sure all tag-keys in coupled_properties are valid inputs with the SCHEMA_TREE
        for coup_tag, coup_key in coupled_properties:
            schema_key = cls.schema_tree.find(coup_tag).find(coup_key)
            if schema_key is None:
                cls.report_warning('cannot couple', tag_key=tag_key, schema_key=(coup_tag, coup_key))
        # the coupled_properties should not already have the property name, then add it to the coupled_properties
        # because they are coupled
        coupled_properties.add(tag_key)

        for coupled_set in cls.sensitivity['coupled']:
            # checking if coupled_properties is subset for any coupled_set that already existed in the class
            # sensitivity; no need to add if yes.
            if coupled_properties.issubset(coupled_set):
                # print("coupled_properties is subset for any coupled_set that already exists")
                return
            # checking if coupled_properties and coupled_set has any intersection; if yes, union them to replace
            # the smaller subsets
            if len(coupled_properties.intersection(coupled_set)):
                # print("coupled_properties and coupled_set has an intersection")
                cls.sensitivity['coupled'].append(coupled_properties.union(coupled_set))
                cls.sensitivity['coupled'].remove(coupled_set)
                return
        # if class coupled sensitivity is empty or neither of the above conditions holds, just add to
        # sensitivity['coupled']
        cls.sensitivity['coupled'].append(coupled_properties)

    def setup_poi(self):
        """ Collects some of the inputs required for the Point of Interconnection (POI) class.

        Returns (Dict): a dictionary of DataFrames and booleans that will be used to initialize the POI

        """
        time_series = self.Scenario['time_series']
        columns = time_series.columns

        poi = {'no_export': self.Scenario['no_export'],
               'no_import': self.Scenario['no_import'],
               'default_growth': self.Scenario['def_growth'],
               'site': None,
               'system': None,
               'aux': None,
               'deferral': None,
               'deferral_growth': None}

        if self.Scenario['incl_site_load'] == 1:
            if 'Site Load (kW)' not in columns:
                e_logger.error('Error: Site load should be included.')
                raise Exception("Missing 'Site Load (kW)' from timeseries input")
            else:
                poi.update({'site': time_series.loc[:, "Site Load (kW)"]})

        return poi

    @classmethod
    def read_referenced_data(cls):
        """ This function makes a unique set of filename(s) based on grab_value_lst.
        It applies for time series filename(s), monthly data filename(s), customer tariff filename(s), and cycle
        life filename(s). For each set, the corresponding class dataset variable (ts, md, ct, cl) is loaded with the data.

        Preprocess monthly data files

        """

        ts_files = cls.grab_value_set('Scenario', 'time_series_filename')
        md_files = cls.grab_value_set('Scenario', 'monthly_data_filename')
        ct_files = cls.grab_value_set('Finance', 'customer_tariff_filename')
        yr_files = cls.grab_value_set('Finance', 'yearly_data_filename')
        cl_files = cls.grab_value_set('Battery', 'cycle_life_filename')

        for ts_file in ts_files:
            cls.referenced_data['time_series'][ts_file] = cls.read_from_file('time_series', ts_file, 'Datetime (he)')
        for md_file in md_files:
            cls.referenced_data['monthly_data'][md_file] = cls.preprocess_monthly(cls.read_from_file('monthly_data', md_file, ['Year', 'Month']))
        for ct_file in ct_files:
            cls.referenced_data['customer_tariff'][ct_file] = cls.read_from_file('customer_tariff', ct_file, 'Billing Period')
        for yr_file in yr_files:
            cls.referenced_data['yearly_data'][yr_file] = cls.read_from_file('yearly_data', yr_file, 'Year')
        for cl_file in cl_files:
            cls.referenced_data['cycle_life'][cl_file] = cls.read_from_file('cycle_life', cl_file)

        return True

    @classmethod
    def grab_value_set(cls, tag, key):
        """ Checks if the tag-key exists in cls.sensitivity, otherwise grabs the base case value
        from cls.template

        Args:
            tag (str):
            key (str):

        Returns: set of values

        """
        try:
            values = set(cls.sensitivity['attributes'][(tag, key)])
        except KeyError:
            try:
                values = {getattr(cls.template, tag)[key]}
            except TypeError:
                values = set()
        return values

    @classmethod
    def build_case_definitions(cls):
        """ Method to create a dataframe that includes all the possible combinations of sensitivity values
        with coupling filter.

        """
        sense = cls.sensitivity["attributes"]
        if not len(sense):
            return

        # apply final coupled filter, remove sets that are not of equal length i.e. select sets only of equal length
        cls.sensitivity['coupled'][:] = [x for x in cls.sensitivity['coupled'] if cls.equal_coupled_lengths(cls.sensitivity['attributes'], x)]
        # determine all combinations of sensitivity analysis without coupling filter
        keys, values = zip(*sense.items())
        all_sensitivity_cases = [dict(zip(keys, v)) for v in itertools.product(*values)]

        case_lst = []
        # sift out sensitivity cases with coupling filter
        for sensitivity_case in all_sensitivity_cases:
            hold = True
            for num, coupled_set in enumerate(cls.sensitivity['coupled']):
                # grab a random property in the COUPLED_SET to know the value length
                rand_prop = next(iter(coupled_set))
                coupled_set_value_length = len(sense[rand_prop])
                # one dictionary of tag-key-value combination of coupled tag-key elements
                coupled_dict = {coupled_prop: sensitivity_case[coupled_prop] for coupled_prop in coupled_set}
                # iterate through possible tag-key-value coupled combos
                for ind in range(coupled_set_value_length):
                    # create a dictionary where the value is the IND-th value of PROP within SENSE and each key is PROP
                    value_combo_dict = {prop: sense[prop][ind] for prop in coupled_set}
                    hold = value_combo_dict == coupled_dict
                    # hold==FALSE -> keep iterating through COUPLED_SET values
                    if hold:
                        # hold==TRUE -> stop iterating through COUPLED_SET values
                        break
                # hold==TRUE -> keep iterating through values of cls.sensitivity['coupled']
                if not hold:
                    # hold==FALSE -> stop iterating through values of cls.sensitivity['coupled']
                    break
            # hold==FALSE -> keep iterating through all_sensitivity_cases
            if hold:
                # hold==TRUE -> add to case_lst & keep iterating through all_sensitivity_cases
                case_lst.append(sensitivity_case)
        cls.case_definitions = pd.DataFrame(case_lst)

    @staticmethod
    def equal_coupled_lengths(value_dic, coupled_set):
        """ Check class sensitivity variable if there is any coupled set has unequal lengths of coupling arrays

            Args:
                value_dic (dict): the dictionary of tag_key pairs that have multiple values
                coupled_set (set): a coupled_set that contains Key/Element tag - Property name of the sensitivity arrays
                being coupled

            Returns:
                True if all those arrays have the same length, the returned set has length of 1 to contain unique value
                False if those arrays have different lengths
        """

        prop_lens = set(map(len, [value_dic[x] for x in coupled_set]))
        if len(prop_lens) == 1:
            return True  # only one unique value length in the set means all list equal length
        else:
            no_cases = np.prod(list(prop_lens))
            message = 'WARNING: coupled sensitivity arrays related to ' + str(list(coupled_set)[0]) \
                      + ' do not have the same length; the total number of cases generated from this coupled_set' \
                      + ' is ' + str(no_cases) + ' because coupling ability for this is ignored.'
            u_logger.debug(message)
            return False

    @staticmethod
    def read_from_file(name, filename, ind_col=None):
        """ Read data from csv or excel file.

        Args:
            name (str): name of data to read
            filename (str): filename of file to read
            ind_col (str or list): column(s) to use as dataframe index

         Returns:
                A pandas dataframe of file at FILENAME location
        """

        raw = pd.DataFrame()

        if (filename is not None) and (not pd.isnull(filename)):

            # logic for time_series data
            parse_dates = name == 'time_series'
            infer_dttm = name == 'time_series'

            # select read function based on file type
            func = pd.read_csv if ".csv" in filename else pd.read_excel

            try:
                raw = func(filename, parse_dates=parse_dates, index_col=ind_col, infer_datetime_format=infer_dttm)
            except UnicodeDecodeError:
                try:
                    raw = func(filename, parse_dates=parse_dates, index_col=ind_col, infer_datetime_format=infer_dttm,
                               encoding="ISO-8859-1")
                except (ValueError, IOError):
                    e_logger.error("Could not open: ", filename)
                    raise ImportError(f"Could not open {name} at '{filename}' from: {os.getcwd()}")
                else:
                    u_logger.info("Successfully read in", filename)
            except (ValueError, IOError):
                e_logger.error("Could not open or could not find: ", filename)
                raise ImportError(f"Could not open or could not find {name} at '{filename}' from: {os.getcwd()}")
            else:
                u_logger.info("Successfully read in: " + filename)

        return raw

    @staticmethod
    def preprocess_monthly(monthly_data):
        """ processing monthly data.
        Creates monthly Period index from year and month

        Args:
            monthly_data (DataFrame): Raw df

        Returns:
            monthly_data (DataFrame): edited df

        """
        if not monthly_data.empty:
            monthly_data.index = pd.PeriodIndex(year=monthly_data.index.get_level_values(0).values,
                                                month=monthly_data.index.get_level_values(1).values, freq='M')
            monthly_data.index.name = 'yr_mo'
            monthly_data.columns = [column.strip() for column in monthly_data.columns]
        return monthly_data

    @classmethod
    def case_builder(cls):
        """
            Function to create all the instances based on the possible combinations of sensitivity values provided
            To update the variable instances of this Params class object

        """
        def load_and_prepare(case_instance, instance_num):
            # local helper function that will call all methods required to prepare the Params instance to be ready to be read into Scenario
            case_instance.load_data_sets()
            if not case_instance.other_error_checks():
                raise Warning(f'Case #{instance_num} has some value error. Please check error log.')
            case_instance.prepare_scenario()
            case_instance.prepare_finance()
            case_instance.prepare_technology()
            case_instance.prepare_services()
            return case_instance

        dictionary = {}
        case = copy.deepcopy(cls.template)
        # while case definitions is not an empty df (there is SA) or if it is the last row in case definitions
        if not cls.case_definitions.empty:
            for index in cls.case_definitions.index:
                row = cls.case_definitions.iloc[index]
                for col in row.index:
                    case.modify_attribute(tupel=col, value=row[col])
                case = load_and_prepare(case, index)
                dictionary.update({index: case})
                case = copy.deepcopy(cls.template)
        else:
            case = load_and_prepare(case, 0)
            dictionary.update({0: case})
        cls.instances = dictionary

    def load_data_sets(self):
        """Assign time series data, cycle life data, and monthly data to Scenario dictionary based on the dataset content type.
           Assign customer tariff data to Finance dictionary based on the customer tariff filename.
           These successful assignments is recorded in the developer log.
        """
        scenario = self.Scenario
        finance = self.Finance
        freq = self.timeseries_frequency(self.Scenario['dt'])
        self.Scenario['frequency'] = freq
        self.Scenario["time_series"] = self.preprocess_timeseries(self.referenced_data["time_series"][scenario["time_series_filename"]], freq)
        self.Scenario["monthly_data"] = self.referenced_data["monthly_data"][scenario["monthly_data_filename"]]
        self.Finance["customer_tariff"] = self.referenced_data["customer_tariff"][finance["customer_tariff_filename"]]
        self.Finance["yearly_data"] = self.referenced_data["yearly_data"][finance["yearly_data_filename"]]
        if self.Battery is not None:
            self.Battery["cycle_life"] = self.referenced_data["cycle_life"][self.Battery["cycle_life_filename"]]

        u_logger.info("Data sets are loaded successfully.")

    @staticmethod
    def timeseries_frequency(dt):
        """ Based on the value of DT, return the frequency that the pandas data time index should take

        Args:
            dt (float): user reported timestep of timeseries dataframe

        Returns: string representation of the frequency of the timeseries index

        """
        # keep only first 3 decimal places
        frequency = ''
        dt_truncated = int(dt * 1000) / 1000
        if dt_truncated == 1.0:
            frequency = 'H'
        if dt_truncated == 0.5:
            frequency = '30min'
        if dt_truncated == .25:
            frequency = '15min'
        if dt_truncated == .083:
            frequency = '5min'
        if dt_truncated == .016:
            frequency = '1min'
        return frequency

    @staticmethod
    def preprocess_timeseries(time_series, freq):
        """ Given a timeseries dataframe, checks to see if the index is timestep begining (starting at
        hour 1) or timestep ending (starting at hour 0). If the timeseries is neither, then error is
        reported to user. Transforms index into timestep beginning.

        Args:
            time_series (DataFrame): raw timeseries dataframe.
            freq (str): frequency that the date time index should take

        Returns: a DataFrame with the index beginning at hour 0

        """
        if time_series.index.hour[0] == 1:
            # set index to denote the start of the timestamp, not the end of the timestamp
            time_series = time_series.sort_index()  # make sure all the time_stamps are in order
            yr_included = time_series.iloc[:-1].index.year.unique()
        elif time_series.index.hour[0] == 0:
            time_series = time_series.sort_index()  # make sure all the time_stamps are in order
            time_series['Start Datetime'] = np.zeros(len(time_series.index))
            yr_included = time_series.index.year.unique()
        else:
            e_logger.error('The timeseries does not start at the begining of the day. Please start with hour as 1 or 0.')
            u_logger.error('The timeseries does not start at the begining of the day. Please start with hour as 1 or 0.')
            raise ValueError('The timeseries does not start at the begining of the day. Please start with hour as 1 or 0.')

        time_series.index = Lib.create_timeseries_index(yr_included, freq)
        time_series.columns = [column.strip() for column in time_series.columns]
        return time_series

    def other_error_checks(self):
        """ Used to collect any other errors that was not specifically detected before.
            The errors is printed in the errors log.
            Including: errors in opt_years, opt_window, validation check for Battery and CAES's parameters.

        Returns (bool): True if there is no errors found. False if there is errors found in the errors log.

        """
        ###################################################################
        # validation that can occur for the class, not just each instance #

        if self.Scenario is None:
            e_logger.error('Please activate the Scenario tag and re-run.')
            return False

        if self.Finance is None:
            e_logger.error('Please activate the Finance tag and re-run.')
            return False

        if self.RA and self.DR:
            e_logger.error('Please pick either Resource Adequacy or Demand Response. They are not compatible with each other.')
            return False

        # require at least 1 energy market participation when participting in any ancillary services
        if (self.SR or self.NSR or self.FR or self.LF) and not (self.retailTimeShift or self.DA):
            e_logger.error('We require energy market participation when participating in ancillary services (ie SR, NSR, FR, LF). +'
                           'Please activate the DA service or the RT service and run again.')
            return False

        ###################################################
        # validation that needs to occur on each instance #

        scenario = self.Scenario
        # storage = None
        retail_time_shift = self.retailTimeShift
        dcm = self.DCM

        start_year = scenario['start_year']
        opt_years = scenario['opt_years']
        time_series_dict = self.referenced_data["time_series"]
        incl_site_load = scenario['incl_site_load']

        for time_series_name in time_series_dict:
            # fix time_series index
            # TODO: what if the user gives timesteps with hour begining? or a
            time_series_index = time_series_dict[time_series_name].index - pd.Timedelta('1s')
            years_included = time_series_index.year.unique()
            if any(value < years_included[0] for value in opt_years):
                e_logger.error("Params Error: The 'opt_years' input starts before the given Time Series.")
                return False

            if all(years_included.values < start_year.year):
                e_logger.error('Params Error: "start_year" is set after the date of the time series file.')
                return False

        if scenario['end_year'].year < start_year.year:
            # make sure that the project start year is before the project end year
            e_logger.error('Params Error: end_year < start_year. end_year should be later than start_year')
            return False

        # validation checks for a Battery's parameters
        if self.Battery:
            # max ratings should not be greater than the min rating for power and energy
            if self.Battery['ch_min_rated'] > self.Battery['ch_max_rated']:
                e_logger.error('Params Error: ch_max_rated < ch_min_rated. ch_max_rated should be greater than ch_min_rated')
                return False

            if self.Battery['dis_min_rated'] > self.Battery['dis_max_rated']:
                e_logger.error('Params Error: dis_max_rated < dis_min_rated. dis_max_rated should be greater than dis_min_rated')
                return False

            if self.Battery['ulsoc'] < self.Battery['llsoc']:
                e_logger.error('Params Error: ulsoc < llsoc. ulsoc should be greater than llsoc')
                return False

            if self.Battery['soc_target'] < self.Battery['llsoc']:
                e_logger.error('Params Error: soc_target < llsoc. soc_target should be greater than llsoc')
                return False

        # validation checks for a CAES's parameters
        if self.CAES:
            if self.CAES['ch_min_rated'] > self.CAES['ch_max_rated']:
                e_logger.error('Params Error: ch_max_rated < ch_min_rated. ch_max_rated should be greater than ch_min_rated')
                return False

            if self.CAES['dis_min_rated'] > self.CAES['dis_max_rated']:
                e_logger.error('Params Error: dis_max_rated < dis_min_rated. dis_max_rated should be greater than dis_min_rated')
                return False

        if (dcm is not None or retail_time_shift is not None) and incl_site_load != 1:
            e_logger.error('Error: incl_site_load should be = 1')
            return False

        if self.RA and self.Scenario['no_export'] or self.RA and self.Scenario['no_import']:
            # if an event is scheduled to occur, then no solution will be found
            e_logger.error('Please do not include POI constraints when participating in Resource Adequacy.')
            return False

        if self.Deferral and self.Scenario['no_export']:
            # if an event is scheduled to occur, then no solution will be found
            e_logger.error('Please do not include the no_export POI constraint when participating in Deferral.')
            return False

        if type(scenario['n']) != str and self.Scenario['n'] > 8764:
            e_logger.error('Params Error: scenario optimization window cannot be greater than 1 year')
            return False
        if type(scenario['n']) != str and scenario['n'] < 2:
            # if N is not a string, then it (the optimization window) cannot be less than 2 hours
            e_logger.error('Params Error: scenario optimization window cannot be less than 2 hours')
            return False
        return True

    def prepare_scenario(self):
        """ Interprets user given data and prepares it for Scenario.

        TODO: add error catching and more explicit messages

        """
        self.POI = self.setup_poi()

        # set dt to be exactly dt
        frequency = self.Scenario['frequency']
        if frequency == '5min':
            self.Scenario['dt'] = 5 / 60
        if frequency == '1min':
            self.Scenario['dt'] = 1 / 60

        # determine if mpc
        # self.Scenario['mpc'] = (self.Scenario['n_control'] != 0)
        self.Scenario['mpc'] = False

        # determine if customer_sided
        self.Scenario['customer_sided'] = (self.DCM is not None) or (self.retailTimeShift is not None)

        u_logger.info("Successfully prepared the Scenario and some Finance")

    def prepare_finance(self):
        """ Interprets user given data and prepares it for Finance.

        """
        # include data in financial class
        self.Finance.update({'n': self.Scenario['n'],
                             'mpc': self.Scenario['mpc'],
                             'start_year': self.Scenario['start_year'],
                             'end_year': self.Scenario['end_year'],
                             'CAES': self.CAES is not None,
                             'dt': self.Scenario['dt'],
                             'opt_years': self.Scenario['opt_years'],
                             'def_growth': self.Scenario['def_growth'],
                             'frequency': self.Scenario['frequency'],
                             'verbose': self.Scenario['verbose'],
                             'customer_sided': self.Scenario['customer_sided']})

        u_logger.info("Successfully prepared the Finance")

    def prepare_technology(self):
        """ Interprets user given data and prepares it for Storage/Storage.

        Returns: collects required timeseries columns required + collects power growth rates

        """
        time_series = self.Scenario['time_series']
        monthly_data = self.Scenario['monthly_data']
        freq = self.Scenario['frequency']

        timeseries_columns = time_series.columns
        monthly_columns = monthly_data.columns

        if self.CAES is not None:
            # check to make sure data was included
            if 'Natural Gas Price ($/MillionBTU)' not in monthly_columns:
                e_logger.error('Error: Please include a monthly natural gas price.')
                raise Exception("Missing 'Natural Gas Price ($/MillionBTU)' from monthly data input")

            # add monthly data and scenario case parameters to CAES parameter dictionary
            self.CAES.update({'natural_gas_price': self.monthly_to_timeseries(freq, self.Scenario['monthly_data'].loc[:, ['Natural Gas Price ($/MillionBTU)']]),
                              'binary': self.Scenario['binary'],
                              'slack': self.Scenario['slack'],
                              'dt': self.Scenario['dt']})
            if self.Scenario['slack']:
                self.CAES.update({'kappa_ene_max': self.Scenario['kappa_ene_max'],
                                  'kappa_ene_min': self.Scenario['kappa_ene_min'],
                                  'kappa_ch_max': self.Scenario['kappa_ch_max'],
                                  'kappa_ch_min': self.Scenario['kappa_ch_min'],
                                  'kappa_dis_max': self.Scenario['kappa_dis_max'],
                                  'kappa_dis_min': self.Scenario['kappa_dis_min']})

        if self.Battery is not None:
            # add scenario case parameters to battery parameter dictionary
            self.Battery.update({'binary': self.Scenario['binary'],
                                 'slack': self.Scenario['slack'],
                                 'dt': self.Scenario['dt']})
            if self.Scenario['slack']:
                self.Battery.update({'kappa_ene_max': self.Scenario['kappa_ene_max'],
                                     'kappa_ene_min': self.Scenario['kappa_ene_min'],
                                     'kappa_ch_max': self.Scenario['kappa_ch_max'],
                                     'kappa_ch_min': self.Scenario['kappa_ch_min'],
                                     'kappa_dis_max': self.Scenario['kappa_dis_max'],
                                     'kappa_dis_min': self.Scenario['kappa_dis_min']})

        if self.PV is not None:
            # check to make sure data was included
            if 'PV Gen (kW/rated kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a PV generation.')
                raise Exception("Missing 'PV Gen (kW/rated kW)' from timeseries input")

            # add time-series data and scenario case parameters to CAES parameter dictionary
            self.PV.update({'rated gen': time_series.loc[:, 'PV Gen (kW/rated kW)'],
                            # 'growth': self.Scenario['def_growth'],
                            'dt': self.Scenario['dt']})

        if self.ICE is not None:
            # add scenario case parameters to diesel parameter dictionary
            self.ICE.update({'dt': self.Scenario['dt']})

        if self.Load is None and self.Scenario['incl_site_load']:
            # Load was not included in MP sheet (so no scalar inputs)
            # AND if the user wants to include load in analysis
            if 'Site Load (kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a site load.')
                raise Exception("Missing 'Site Load (kW)' from timeseries input")
            self.Load = {'name': "Site Load",
                         'power_rating': 0,
                         'duration': 0,
                         'nsr_response_time': 0,
                         'sr_response_time': 0,
                         'startup_time': 0,
                         'dt': self.Scenario['dt'],
                         'growth': self.Scenario['def_growth'],
                         'site_load': time_series.loc[:, 'Site Load (kW)']}

        u_logger.info("Successfully prepared the Technologies")

    @staticmethod
    def monthly_to_timeseries(freq, column):
        """ Converts data given monthly into timeseries.

        Args:
            freq (str): frequency that the date time index should take
            column (DataFrame):  A column of monthly data given by the user, indexed by month

        Returns: A Series of timeseries data that is equivalent to the column given

        """
        first_day = f"1/1/{column.index.year[0]}"
        last_day = f"1/1/{column.index.year[-1]+1}"

        new_index = pd.date_range(start=first_day, end=last_day, freq=freq, closed='left')
        temp = pd.DataFrame(index=new_index)
        temp['yr_mo'] = temp.index.to_period('M')
        temp = temp.reset_index().merge(column.reset_index(), on='yr_mo', how='left').set_index(new_index)
        # temp = temp.drop(columns=['yr_mo'])
        return temp[column.columns[0]]

    def prepare_services(self):
        """ Interprets user given data and prepares it for each ValueStream (dispatch and pre-dispatch).

        Returns: collects required power timeseries

        """
        monthly_data = self.Scenario['monthly_data']
        time_series = self.Scenario['time_series']
        dt = self.Scenario['dt']
        timeseries_columns = time_series.columns
        monthly_columns = monthly_data.columns
        freq = self.Scenario['frequency']

        if self.Deferral is not None:
            if 'Deferral Load (kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a monthly fuel price.')
                raise Exception("Missing 'Deferral Load (kW)' from timeseries input")
            self.POI.update({'deferral': time_series.loc[:, "Deferral Load (kW)"],
                             'deferral_growth': self.Deferral['growth']})
            self.Deferral["last_year"] = self.Scenario["end_year"]
            self.Deferral.update({'load': time_series.loc[:, 'Deferral Load (kW)']})

        if self.Volt is not None:
            if 'Vars Reservation (%)' not in timeseries_columns:
                e_logger.error('Error: Please include the vars reservation as percent (from 0 to 100) of inverter max.')
                raise Exception("Missing 'Vars Reservation (%)' from timeseries input")
            # make sure that vars reservation inputs are between 0 and 100
            percent = time_series.loc[:, 'VAR Reservation (%)']
            if len([*filter(lambda x: x >= 30, percent.value)]):
                e_logger.error('Error: Please include the vars reservation as percent (from 0 to 100) of inverter max.')
                raise Exception("Value error within 'Vars Reservation (%)' timeseries")
            self.Volt['percent'] = percent

        included_system_load = False

        if self.RA is not None:
            if 'active hours' in self.RA['idmode'].lower():
                # if using active hours, require the column
                if 'RA Active (y/n)' not in timeseries_columns:
                    e_logger.error('Error: Please include when RA is active.')
                    raise Exception("Missing 'RA Active (y/n)' from timeseries input")
                self.RA['active'] = time_series.loc[:, 'RA Active (y/n)']
            if 'RA Capacity Price ($/kW)' not in monthly_columns:
                e_logger.error('Error: Please include monthly RA price.')
                raise Exception("Missing 'RA Price ($/kW)' from monthly CSV input")

            if 'System Load (kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a system load.')
                raise Exception("Missing 'System Load (kW)' from timeseries input")
            included_system_load = True  # flag so we dont check for system load again
            self.RA.update({'value': monthly_data.loc[:, 'RA Capacity Price ($/kW)'],
                            'default_growth': self.Scenario['def_growth'],  # applied to system load if 'forecasting' load
                            'system_load': time_series.loc[:, "System Load (kW)"],
                            'active': time_series.loc[:, 'RA Active (y/n)'],
                            'dt': dt})

        if self.DR is not None:
            if 'DR Capacity Price ($/kW)' not in monthly_columns:
                e_logger.error('Error: Please include DR capcity prices.')
                raise Exception("Missing 'DR Capacity Price ($/kW)' from monthly input")
            if 'DR Energy Price ($/kWh)' not in monthly_columns:
                e_logger.error('Error: Please include DR energy prices.')
                raise Exception("Missing 'DR Energy Price ($/kWh)' from monthly input")
            if 'DR Months (y/n)' not in monthly_columns:
                e_logger.error('Error: Please include DR months.')
                raise Exception("Missing 'DR Months (y/n)' from monthly input")
            if 'DR Capacity (kW)' not in monthly_columns:
                e_logger.error('Error: Please include a DR capacity.')
                raise Exception("Missing 'DR Capacity (kW)' from monthly input")
            if 'Site Load (kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a site load.')
                raise Exception("Missing 'Site Load (kW)' from timeseries input")

            # self.Scenario['incl_site_load'] = 1

            # if we havent checked for a system load already, check now
            if not included_system_load and 'System Load (kW)' not in timeseries_columns:
                e_logger.error('Error: Please include a system load.')
                raise Exception("Missing 'System Load (kW)' from timeseries input")
            self.DR.update({'cap_price': monthly_data.loc[:, 'DR Capacity Price ($/kW)'],
                            'ene_price': monthly_data.loc[:, 'DR Energy Price ($/kWh)'],
                            'dr_months': self.monthly_to_timeseries(freq, monthly_data.loc[:, ['DR Months (y/n)']]),
                            'dr_cap': self.monthly_to_timeseries(freq, monthly_data.loc[:, ['DR Capacity (kW)']]),
                            'cap_monthly': monthly_data.loc[:, 'DR Capacity (kW)'],
                            'default_growth': self.Scenario['def_growth'],
                            'system_load': time_series.loc[:, "System Load (kW)"],
                            'site_load': time_series.loc[:, "Site Load (kW)"],
                            'dt': dt})

        if included_system_load:
            self.POI.update({'system': time_series.loc[:, "System Load (kW)"]})

        if self.Backup is not None:
            if 'Backup Price ($/kWh)' not in monthly_columns:
                e_logger.error('Error: Please include Backup Price in monthly data csv.')
                raise Exception("Missing 'Backup Price ($/kWh)' from monthly data input")
            if 'Backup Energy (kWh)' not in monthly_columns:
                e_logger.error('Error: Please include Backup Energy in monthly data csv.')
                raise Exception("Missing 'Backup Energy (kWh)' from monthly data input")
            self.Backup.update({'daily_energy': self.monthly_to_timeseries(freq, monthly_data.loc[:, ['Backup Energy (kWh)']]),
                                'monthly_price': monthly_data.loc[:, 'Backup Price ($/kWh)'],
                                'monthly_energy': monthly_data.loc[:, 'Backup Energy (kWh)']})

        if self.SR is not None:
            self.SR.update({'price': time_series.loc[:, 'SR Price ($/kW)']})

        if self.NSR is not None:
            self.NSR.update({'price': time_series.loc[:, 'NSR Price ($/kW)']})

        if self.DA is not None:
            self.DA.update({'price': time_series.loc[:, 'DA Price ($/kWh)']})

        if self.FR is not None:
            if self.FR['CombinedMarket']:
                self.FR.update({'regu_price': np.divide(time_series.loc[:, 'FR Price ($/kW)'], 2),
                                'regd_price': np.divide(time_series.loc[:, 'FR Price ($/kW)'], 2),
                                'energy_price': time_series.loc[:, 'DA Price ($/kWh)']})
            else:
                self.FR.update({'regu_price': time_series.loc[:, 'Reg Up Price ($/kW)'],
                                'regd_price': time_series.loc[:, 'Reg Down Price ($/kW)'],
                                'energy_price': time_series.loc[:, 'DA Price ($/kWh)']})

        if self.LF is not None:
            if self.LF['CombinedMarket']:
                self.LF.update({'lf_up_price': np.divide(time_series.loc[:, 'LF Price ($/kW)'], 2),
                                'lf_do_price': np.divide(time_series.loc[:, 'LF Price ($/kW)'], 2),
                                'lf_offset': time_series.loc[:, 'LF Offset (kW)'],
                                'energy_price': time_series.loc[:, 'DA Price ($/kWh)']})
            else:
                self.LF.update({'lf_up_price': time_series.loc[:, 'LF Up Price ($/kW)'],
                                'lf_do_price': time_series.loc[:, 'LF Down Price ($/kW)'],
                                'lf_offset': time_series.loc[:, 'LF Offset (kW)'],
                                'energy_price': time_series.loc[:, 'DA Price ($/kWh)']})

        if self.User is not None:
            # check to make sure the user included at least one of the custom constraints
            input_cols = ['Power Max (kW)', 'Power Min (kW)', 'Energy Max (kWh)', 'Energy Min (kWh)']
            if not time_series.columns.isin(input_cols).any():
                e_logger.error("User has inputted an invalid constraint for Storage. Please change and run again.")
                raise Exception("User has inputted an invalid constraint for Storage. Please change and run again.")
            self.User.update({'constraints': time_series.loc[:, list(np.intersect1d(input_cols, list(time_series)))]})

        if self.Scenario['customer_sided']:
            retail_prices = Financial.calc_retail_energy_price(self.Finance['customer_tariff'], self.Scenario['frequency'], self.Scenario['opt_years'])
            if self.retailTimeShift is not None:
                self.retailTimeShift.update({'price': retail_prices.loc[:, 'p_energy'],
                                             'tariff': self.Finance['customer_tariff'].loc[
                                                       self.Finance['customer_tariff'].Charge.apply((lambda x: x.lower())) == 'energy', :]})

            if self.DCM is not None:
                self.DCM.update({'tariff': self.Finance['customer_tariff'].loc[
                                           self.Finance['customer_tariff'].Charge.apply((lambda x: x.lower())) == 'demand', :],
                                 'billing_period': retail_prices.loc[:, 'billing_period']})

        u_logger.info("Successfully prepared the value-stream (services)")

    def modify_attribute(self, tupel, value):
        """ Function to modify the value of the sensitivity value based on the specific instance/scenario

            Args:
                tupel (tuple): the group of attributes
                value (int/float): the new value to replace the former value

        """

        attribute = getattr(self, tupel[0])
        attribute[tupel[1]] = value

    @classmethod
    def storagevet_requirement_check(cls):
        """ Used to test only requirements that don't apply for DERVET
            The errors is printed in the errors log.

        Returns (bool): False if there occurs errors.

        """
        if cls.template.CAES is not None and cls.template.Battery is not None:
            e_logger.error("Storage technology CAES and Battery should not be active together in StorageVET.")
            return False

        return True
