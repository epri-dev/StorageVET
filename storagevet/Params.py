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
Params.py

"""
import copy
import itertools
import json
import xml.etree.ElementTree as eT
from datetime import datetime
from shutil import copy2

import numpy as np
import pandas as pd

from storagevet.ErrorHandling import *
from storagevet.Finances import Financial
from storagevet.Library import create_timeseries_index, is_leap_yr, truncate_float


class Params:
    """
        Class attributes are made up of services, technology, and any other needed inputs. The
        attributes are filled by converting the json file in a python object.

        Notes:
             Might need to modify the summary functions for pre-visualization every time the Params
             class is changed in terms of the naming of the class variables, the active
             components and their dictionary structure - TN
    """
    # set schema loction based on the location of this file
    schema_location = Path(__file__).absolute().with_name('schema.json')

    # initialize class variables
    schema_dct = None
    json_tree = None
    xmlTree = None
    filename = None

    active_tag_key = None
    sensitivity = None  # contains all sensitivity variables as keys and their values as lists
    case_definitions = None  # Each row specifies the value of each attribute

    instances = None
    template = None
    referenced_data = None  # for a scenario in the sensitivity analysis

    results_inputs = None

    input_error_raised = False
    timeseries_missing_error = False
    timeseries_data_error = False
    monthly_error = False

    @classmethod
    def initialize(cls, filename, verbose):
        """ This function 1) converts CSV into JSON; 2) read in the JSON; 3) convert indirect data
        references into direct data; 4) create instances from Sensitivity Analysis and Coupling
        values

            Args:
                filename (string): filename of JSON or CSV model parameter
                verbose (bool): whether or not to print to console for more feedback

            Returns dictionary of instances of Params, each key is a number
        """
        # 1) INITIALIZE CLASS VARIABLES
        cls.input_error_raised = False
        cls.timeseries_missing_error = False
        cls.timeseries_data_error = False
        cls.monthly_error = False
        # SENSITIVITY contains all sensitivity variables as keys and their values as lists
        cls.sensitivity = {"attributes": dict(),
                           "coupled": list()}
        # CASE DEFINITIONS each row specifies the value of each attribute for sensitivity analysis
        cls.case_definitions = pd.DataFrame()
        cls.instances = dict()
        # holds the data of all the time series usd in sensitivity analysis
        cls.referenced_data = {
            "time_series": dict(),
            "monthly_data": dict(),
            "customer_tariff": dict(),
            "cycle_life": dict(),
            "yearly_data": dict()}

        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        cls.results_inputs = {'label': '',
                              'dir_absolute_path': Path('Results') / timestamp_now,
                              'errors_log_path': Path('Results') / timestamp_now}
        cls.active_tag_key = set()

        # load schema (used to validate data sheets)
        cls.schema_dct = json.load(open(cls.schema_location))
        cls.schema_dct = cls.schema_dct.get("schema")
        if cls.schema_dct is None:
            raise Exception("Schema can not be imported.")

        # 2) CONVERT CSV INTO JSON
        model_param_input_was_csv = False
        filename = Path(filename)
        if '.csv' == filename.suffix:
            model_param_input_was_csv = True
            filename = cls.csv_to_json(filename)

        cls.filename = filename
        # 3) LOAD DIRECT DATA FROM JSON/XML
        if '.json' == filename.suffix:
            cls.json_tree = json.load(open(filename))
            cls.json_tree = cls.json_tree.get("tags")
            if cls.json_tree is None:
                raise Exception("Given model parameter is empty.")
        elif '.xml' == filename.suffix:
            cls.xmlTree = eT.parse(filename)
        else:
            raise ModelParameterError("Please indicate a CSV, XML, or JSON to read the case from.")

        # read Results tag and set as class attribute
        # NOTE: sensitivity analysis does not work on the 'Results' tag. this is by design.
        result_dic = cls.flatten_tag_id(cls.read_and_validate('Results'))
        if result_dic:
            cls.results_inputs = result_dic
            cls.results_inputs['errors_log_path'] = Path(cls.results_inputs['errors_log_path'])
        TellUser.create_log(cls.results_inputs['errors_log_path'], verbose)

        # copy the model parameters csv/json into the Results folder
        # TODO: expand all paths to full in these objects before copying?
        #        (so that they become usable regardless of where run_DERVET.py is called from)
        cls.copy_model_params_to_results(model_param_input_was_csv)

        # _init_ the Params class
        cls.template = cls()

        # report back any warning
        if cls.input_error_raised or cls.bad_active_combo():
            #TellUser.close_log()
            raise ModelParameterError(
                "The model parameter has some errors associated to it. Please fix and rerun.")

        # turn referenced data into direct data and preprocess some
        cls.read_referenced_data()

        # 4) SET UP SENSITIVITY ANALYSIS AND COUPLING W/ CASE BUILDER
        TellUser.info('Updating data for Sensitivity Analysis...')

        # build each of the cases that need to be run based on the model parameter inputed
        cls.build_case_definitions()
        cls.case_builder()
        cls.load_and_prepare()

        return cls.instances

    @classmethod
    def copy_model_params_to_results(cls, csv_was_used):
        # copy the model parameters file that the code uses (.json or .xml)
        src = cls.filename
        if src.is_file():
            dst = cls.results_inputs['errors_log_path'] / f'model_parameters{src.suffix}'
            copy2(src, dst)
            TellUser.info(f'JSON Model Parameter ({src}) was copied to Results ({dst})')
        if csv_was_used:
            # also copy the CSV model parameters file, if there is one in use
            src = Path(str(src.parent / src.stem) + '.csv')
            if src.is_file():
                dst = cls.results_inputs['errors_log_path'] / f'model_parameters{src.suffix}'
                copy2(src, dst)
                TellUser.info(f'CSV Model Parameter ({src}) was copied to Results ({dst})')

    @classmethod
    def csv_to_json(cls, csv_filepath):
        """ converts a CSV with the correct set of columns into a json

        Args:
            csv_filepath:

        Returns:

        """
        # open csv to read into dataframe
        csv_data = pd.read_csv(csv_filepath)
        # convert pandas df into DER-VET readable data structure
        input_dct = cls.pandas_to_dict(csv_data)
        # find .csv in the filename and replace with .json
        json_filename = csv_filepath.with_suffix('.json')
        # dump json at original file location
        with open(json_filename, 'w') as json_file:
            json.dump(input_dct, json_file, sort_keys=True, indent=4)
        return json_filename

    @staticmethod
    def pandas_to_dict(model_parameter_pd):
        """converts csv to a json--which DERVET can then read directly

        Args:
            model_parameter_pd:

        Returns: dictionary that can be jumped as json in the data structure that DER-VET reads

        """
        # check if there was an ID column, if not then add one filled with '.'
        if 'ID' not in model_parameter_pd.columns:
            model_parameter_pd['ID'] = np.repeat('', len(model_parameter_pd))
        # write the header of the json file
        input_dct = {
            "name": str(None),
            "type": "Expert",
            "tags": dict()
        }
        tag_dict = input_dct["tags"]
        # specify columns to place
        key_attributes = set(model_parameter_pd.columns) - {'Tag', 'Key', 'ID', 'Units',
                                                            'Allowed Values', 'Description',
                                                            'Active',
                                                            'Evaluation Value',
                                                            'Evaluation Active',
                                                            'Sensitivity Parameters', 'Coupled',
                                                            'Options/Notes'}
        # key_attributes = ['Optimization Value', 'Value', 'Type', 'Sensitivity Analysis']
        # outer loop for each tag/object and active status, i.e. Scenario, Battery, DA, etc.
        for tag in model_parameter_pd.Tag.unique():
            # add tag to TAG_DICT and initialize dict
            tag_dict[tag] = dict()
            # select all TAG rows
            tag_sub = model_parameter_pd.loc[model_parameter_pd.Tag == tag]
            # loop through each unique value in ID
            for id_str in tag_sub.ID.unique():
                # select rows with given ID_STR
                id_tag_sub = tag_sub.loc[tag_sub.ID == id_str]
                # add id to dictionary for TAG
                key_tag_id_dict = dict()
                tag_dict[tag][str(id_str)] = {
                    "active": id_tag_sub.Active.iloc[0].strip(),
                    "keys": key_tag_id_dict
                }
                # middle loop for each object's elements and is sensitivity is needed
                for ind, row in id_tag_sub.iterrows():
                    if row['Key'] is np.nan:
                        continue
                    key_tag_id_dict[row['Key'].strip()] = dict()
                    attr_key_tag_id_dict = key_tag_id_dict[row['Key'].strip()]
                    # inner loop for specifying elements Value, Type, Coupled, etc.
                    for attr in key_attributes:
                        attr_key = attr.strip().lower()
                        if attr_key in ['optimization value', 'value']:
                            attr_key = 'opt_value'
                        if attr_key == 'sensitivity analysis':
                            attr_key = 'sensitivity'
                            attr_value = {
                                "active": str(row[attr]),
                                "value": str(row["Sensitivity Parameters"]),
                                "coupled": str(row["Coupled"])
                            }
                        else:
                            attr_value = str(row[attr])
                        attr_key_tag_id_dict[attr_key] = attr_value
        return input_dct

    def __init__(self):
        """ Initialize these following attributes of the empty Params class object.

        """
        self.Scenario = self.read_and_validate('Scenario')
        self.Finance = self.read_and_validate('Finance')

        # value streams
        self.DA = self.read_and_validate('DA')
        self.FR = self.read_and_validate('FR')
        self.SR = self.read_and_validate('SR')
        self.NSR = self.read_and_validate('NSR')
        self.LF = self.read_and_validate('LF')
        self.DCM = self.read_and_validate('DCM')
        self.retailTimeShift = self.read_and_validate('retailTimeShift')
        self.Backup = self.read_and_validate('Backup')  # this is an empty dictionary
        self.Deferral = self.read_and_validate('Deferral')
        self.User = self.read_and_validate('User')
        self.DR = self.read_and_validate('DR')
        self.RA = self.read_and_validate('RA')
        self.Volt = self.read_and_validate('Volt')

        # technologies/DERs
        self.PV = self.read_and_validate('PV')
        self.Battery = self.read_and_validate('Battery')
        self.CAES = self.read_and_validate('CAES')
        self.ICE = self.read_and_validate('ICE')
        # attributes to be set after datasets are loaded and all instances are made
        self.POI = None
        self.Load = None

    @classmethod
    def read_and_validate(cls, name):
        """ Read data from a file (usually after conversion from model parameter CSV) and
        initializes senesitivity analysis. Makes note of which Key in what Tag has sensitivity
        analysis by assigning a tuple of the key and property name as the index to the
        sensitivity values given. This is stored in the attributes index of the sensitivity
        dictionary, which is a nested dictionary whose first index is a string and second index
        is a tuple

        Args:
            name (str): name of the root 'Tag'

        Returns: A dictionary where keys are the ID value and the key is--
            A dictionary filled with needed given values of the properties of each
            service/technology if the service or technology is not implemented in this case,
            then empty dictionary is returned.
        """
        if '.json' == cls.filename.suffix:
            return cls.read_and_validate_json(name)
        if '.xml' == cls.filename.suffix:
            # TODO: feel free to remove this feature if we need to change read_and_validate
            return cls.read_and_validate_xml(name)

    @classmethod
    def read_and_validate_xml(cls, name):
        """ Read data from xml file (usually after conversion from model parameter CSV) and
        initializes sensitivity analysis. Makes note of which Key in what Tag has sensitivity
        analysis by assigning a tuple of the key and property name as the index to the
        sensitivity values given. This is stored in the attributes index of the sensitivity
        dictionary, which is a nested dictionary whose first index is a string and second index
        is a tuple

        Args:
            name (str): name of the root 'Tag' in the xml file

        Returns: A dictionary where keys are the ID value and the key is--  filled with needed
        given values of the properties of each service/technology if the service or technology
        is not implemented in this case, then empty dictionary is returned.
        """
        schema_tag = cls.schema_dct.get("tags").get(name)
        if schema_tag is None:
            # ignore tags not in schema
            return

        # check how many subelements are allowed per TAG
        max_num = schema_tag.get('max_num')
        tag_elems = cls.xmlTree.findall(name)
        # check to see if user includes the tag within the provided xml
        if tag_elems is None:
            return
        if max_num is not None:
            if int(max_num) < len(tag_elems):
                # report to the user that too many elements for the given TAG were included
                cls.report_warning("too many tags", tag=name, max=max_num, length=len(tag_elems))
                return
        master_dictionary = {}
        for tag in tag_elems:
            # Checks if the first character is 'y' or '1', if true it creates a dictionary.
            if tag.get('active')[0].lower() in ["y", "1"]:
                id_str = tag.get('id')
                tag_dict = {}
                # iterate through each tag required by the schema
                schema_key_dict = schema_tag.get("keys")
                for schema_key_name, schema_key_attr in schema_key_dict.items():
                    # Check if attribute is in the schema
                    try:
                        key = tag.find(schema_key_name)
                        value = key.find('Value').text
                    except (KeyError, AttributeError):
                        cls.report_warning("missing key", tag=name, key=schema_key_name)
                        continue
                    # check to see if input is optional
                    if value == 'nan' and schema_key_attr.get(
                            'optional') != 'y' and schema_key_attr.get('optional') != '1':
                        cls.report_warning("not allowed", tag=name, key=schema_key_name,
                                           value=value,
                                           allowed_values="Something other than 'nan'")
                        continue
                    # check if value is None
                    if value is None:
                        cls.report_warning("not allowed", tag=name, key=schema_key_name,
                                            value=value,
                                            allowed_values="Something other than 'None' or 'null'")
                        continue
                    # convert to correct data type
                    intended_type = schema_key_attr.get('type')
                    # fills dictionary with the base case (this will serve as the template case)
                    base_values = cls.convert_data_type(value, intended_type)
                    # validate the inputted value
                    cls.checks_for_validate(base_values, schema_key_attr, schema_key_name,
                                            f"{name}-{id_str}")  # we want to report the tag & ID
                    # save value to base case
                    tag_dict[schema_key_name] = base_values
                    # if yes, assign this to the class's sensitivity variable.
                    if key.get('analysis')[0].lower() == "y" or key.get('analysis')[0] == "1":
                        tag_key = (name, key.tag, id_str)
                        sensitivity_values = cls.extract_data(
                            key.find('Sensitivity_Parameters').text, intended_type)
                        # validate each value
                        for values in sensitivity_values:
                            label = f"{name}-{id_str}"  # we want to report the tag & ID
                            cls.checks_for_validate(values, schema_key_attr, schema_key_name,
                                                    label)
                        # save all values to build into cases later
                        cls.sensitivity['attributes'][tag_key] = sensitivity_values
                        # check if the property should be Coupled
                        if key.find('Coupled').text != 'None':
                            # get the tag-ID-key set that is coupled with this one
                            coupled_set = cls.parse_coupled(name, id_str, key.find('Coupled').text)
                            cls.fetch_coupled(coupled_set, tag_key)
                # after reading and validating all keys in the ID set, save to master dictionary
                master_dictionary[id_str] = tag_dict
        return master_dictionary

    @classmethod
    def read_and_validate_json(cls, name):
        """ Read data from json file (usually after conversion from model parameter CSV) and
        initializes sensitivity analysis. Makes note of which Key in what Tag has sensitivity
        analysis by assigning a tuple of the key and property name as the index to the
        sensitivity values given. This is stored in the attributes index of the sensitivity
        dictionary, which is a nested dictionary whose first index is a string and second index
        is a tuple

        Args:
            name (str): name of the root 'Tag' in the json file

        Returns: A dictionary where keys are the ID value and the key is-- filled with needed
        given values of the properties of each service/technology if the service or technology
        is not implemented in this case, then empty dictionary is returned.
        """
        schema_tag = cls.schema_dct.get("tags").get(name)
        # Check if tag is in schema (SANITY CHECK)
        if schema_tag is None:
            return
        # check to see if user includes the tag within the provided json
        user_tag = cls.json_tree.get(name)
        if user_tag is None:
            return {}
        # check how many sub-elements are allowed per TAG
        max_num = schema_tag.get('max_num')
        if max_num is not None:
            if int(max_num) < len(user_tag):
                # report to the user that too many elements for the given TAG were included
                cls.report_warning("too many tags", tag=name, max=max_num, length=len(user_tag))
                return
        master_dictionary = {}
        for tag_id, tag_attrib in user_tag.items():
            # Checks if the first character is 'y' or '1', if true it creates a dictionary.
            active_tag = tag_attrib.get('active')
            if active_tag is not None and (active_tag[0].lower() in ["y", "1"]):
                dictionary = {}
                # grab the user given keys
                user_keys = tag_attrib.get('keys')
                # iterate through each key required by the schema
                schema_key_dict = schema_tag.get("keys")
                for schema_key_name, schema_key_attr in schema_key_dict.items():
                    # Check if attribute is in the schema
                    try:
                        key = user_keys.get(schema_key_name)
                        value = key.get('opt_value')
                    except (KeyError, AttributeError):
                        cls.report_warning("missing key", tag=name, key=schema_key_name)
                        continue
                    # check to see if input is optional
                    if value == 'nan':
                        optional = schema_key_attr.get('optional')
                        if optional is not None and (optional[0].lower() in ["y", "1"]):
                            # value does not have to be included (don't error)
                            pass
                        elif optional is not None:
                            # check if the value corresponding value is not nan
                            corresponding_key = user_keys.get(optional, 'nan')
                            corresponding_key_value = corresponding_key.get('opt_value')
                            if corresponding_key_value == 'nan':
                                # corresponding key was not given a value, so error
                                cls.report_warning("not allowed", tag=name, key=schema_key_name,
                                                   value=value,
                                                   allowed_values="Something other than 'nan'. " +
                                                                  f" Or define {optional}")
                        else:
                            # value cannot be nan (error)
                            cls.report_warning("not allowed", tag=name, key=schema_key_name,
                                               value=value,
                                               allowed_values="Something other than 'nan'")
                        continue
                    # check if value is None
                    if value is None:
                        cls.report_warning("not allowed", tag=name, key=schema_key_name,
                                            value=value,
                                            allowed_values="Something other than 'None' or 'null'")
                        continue
                    # convert to correct data type
                    intended_type = schema_key_attr.get('type')
                    # fills dictionary with base case from the json file (serves as the template)
                    base_values = cls.convert_data_type(value, intended_type)
                    # validate the inputted value
                    cls.checks_for_validate(base_values, schema_key_attr, schema_key_name,
                                            f"{name}-{tag_id}")  # we want to report the tag & ID
                    # save value to base case
                    dictionary[schema_key_name] = base_values
                    # if yes, assign this to the class's sensitivity variable.
                    key_sensitivity = key.get('sensitivity')
                    if key_sensitivity is not None:
                        key_sensitivity_active = key_sensitivity.get('active')
                        if key_sensitivity_active is not None and \
                                (key_sensitivity_active[0].lower() in ["y","1"]):
                            tag_key_id = (name, schema_key_name, tag_id)
                            # parse the values for sensitivity analysis
                            unparsed_sensitivity_values = key_sensitivity.get('value')
                            if unparsed_sensitivity_values is None:
                                continue
                                # todo report back to user:
                                #   "sensitivity analysis is active, but no values were defined"
                            sensitivity_values = cls.extract_data(unparsed_sensitivity_values,
                                                                  intended_type)
                            # validate each value
                            for values in sensitivity_values:
                                cls.checks_for_validate(values, schema_key_attr, schema_key_name,
                                                        f"{name}-{tag_id}")
                            # save all values to build into cases later
                            cls.sensitivity['attributes'][tag_key_id] = sensitivity_values
                            # check if the property should be Coupled
                            coupled_attribute = key_sensitivity.get('coupled')
                            if coupled_attribute is not None and coupled_attribute != 'None':
                                coupled_with = cls.parse_coupled(name, tag_id, coupled_attribute)
                                cls.fetch_coupled(coupled_with, tag_key_id)
                # after reading & validating keys in the ID set, save to the master dictionary
                master_dictionary[tag_id] = dictionary
        return master_dictionary

    @staticmethod
    def convert_data_type(value, desired_type):
        """ coverts data to a given type. takes in a value and a type

            Args:
                value (str): some data needed to be converted
                desired_type (str): the desired type of the given value

            Returns:
                Attempts to convert the type of 'value' to 'desired_type'. Returns value if it can,
                otherwise it returns a tuple (None, value) if desired_type is not a known type to
                the function or the value cannot be converted to the desired type. report if
                there was an error converting the given data in the convert_data_type method

            Notes if None or NaN is given for the value of VALUE, then this function will error
        """

        if desired_type == 'string':
            return value.lower()
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
                return value.lower()
        elif desired_type == 'list/string':
            try:
                return value.split()
            except ValueError:
                return None, value
        else:
            return None, value

    @classmethod
    def checks_for_validate(cls, value, schema_key_attr, key_name, tag_name):
        """ Helper function to validate method. This runs the checks to validate the value of input
        data with its properties (the arguments) provided from Schema.json

            Args:
                value: the value of the specific key being looked at, as read in from the json
                schema_key_attr (dict): schema key for the value being validated
                key_name (str): the key in question
                tag_name (str): the tag from which this key is organized in

            Returns: error as a tuple of information (tag, key, what caused the error, the type of
            error, what the error should be changed into) if no error, then return None

        """

        desired_type = schema_key_attr.get('type')
        # check if there was a conversion error
        if type(value) == tuple:
            cls.report_warning('conversion', value=value[1], type=desired_type, key=key_name,
                               tag=tag_name)
            return

        # check if the data belongs to a set of allowed values
        allowed_values = schema_key_attr.get('allowed_values')
        if allowed_values is not None:
            allowed_values = {cls.convert_data_type(item, desired_type) for item in
                              allowed_values.split('|')}  # make a list of the allowed values
            # if the data type is 'STRING', then convert to lower case, then compare
            if desired_type == 'string':
                value = value.lower()
                allowed_values = {check_value.lower() for check_value in allowed_values}
            if not len({value}.intersection(allowed_values)):
                # check whether to consider a numerical bound for the set of possible values
                if 'bound' not in allowed_values:
                    cls.report_warning('not allowed', value=value, allowed_values=allowed_values,
                                       key=key_name, tag=tag_name)
                    return

        # check if data is in valid range
        bound_error = False
        minimum = schema_key_attr.get('min')
        if minimum is not None:
            minimum = cls.convert_data_type(minimum, desired_type)
            try:  # handle if minimum is None
                bound_error = bound_error or value < minimum
            except TypeError:
                pass
        maximum = schema_key_attr.get('max')
        if maximum is not None:
            maximum = cls.convert_data_type(maximum, desired_type)
            try:  # handle if maximum is None
                bound_error = bound_error or value > maximum
            except TypeError:
                pass
        if bound_error:
            cls.report_warning('size', value=value, key=key_name, tag=tag_name, min=minimum,
                               max=maximum)
        return

    @classmethod
    def report_warning(cls, warning_type, raise_input_error=True, **kwargs):
        """ Print a warning to the user log. Warnings are reported, but do not result in exiting.

        Args:
            warning_type (str): the classification of the warning to be reported to the user
            raise_input_error (bool): raise this warning as an error instead back to the user and
                stop running the program
            kwargs: elements about the warning that need to be reported to the user (like the tag
                and key that caused the error

        """
        if warning_type == "unknown tag":
            TellUser.warning(
                f"INPUT: {kwargs['tag']} is not used in this program and will be ignored. " +
                "Please check spelling and rerun if you would like the key's associated to be " +
                "considered.")

        if warning_type == "missing key":
            TellUser.error(f"INPUT: {kwargs['tag']}-{kwargs['key']} is not found in model " +
                           "parameter. Please include and rerun.")

        if warning_type == "conversion":
            TellUser.error(f"INPUT: error converting {kwargs['tag']}-{kwargs['key']}" +
                           f"...trying to convert '{kwargs['value']}' into {kwargs['type']}")

        if warning_type == 'size':
            TellUser.error(
                f"INPUT: {kwargs['tag']}-{kwargs['key']}  value ('{kwargs['value']}') " +
                f"is out of bounds. Max: ({kwargs['max']}  Min: {kwargs['min']})")

        if warning_type == "not allowed":
            TellUser.error(
                f"INPUT: {kwargs['tag']}-{kwargs['key']}  value ('{kwargs['value']}') is not a " +
                "valid input. The value should be one of the following: " +
                f"{kwargs['allowed_values']}")

        if warning_type == "cannot couple":
            TellUser.error(f"INPUT: Error occurred when trying to couple {kwargs['tag_key']} " +
                           f"with {kwargs['schema_key']}. Please fix and run again.")

        cls.input_error_raised = cls.input_error_raised or raise_input_error

    @classmethod
    def extract_data(cls, expression, data_type):
        """
            Function to extract out all the sensitivity or non-sensitivity values for each given
            expression, due to their string format. And then to convert these data values to the
            proper data types in prior to the return.

            Args:
                expression (str): the string of expression to be extracted
                data_type (str): the string of dataType

            Returns:
                result (list): the list with each element as a sensitivity value type-casted for
                analysis. in the case of non-sensitivity this list is only one element in size.

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
    def parse_coupled(tag, id_str, unparsed_coupled_with):
        """ Function to allow coupling across different types of Tags (Scenario, Finance, etc) by
        improving text parsing. By filtering out the types (categories) and then mapping them to
        the respective property result. Create a list that contains coupled_sets of properties
        from different types or same type. Improve readability so that fetch_coupled can be done
        appropriately.

            Args:
                tag (str): string representation of tag of csv (top level of the json)
                id_str (str): the ID string for the tag
                unparsed_coupled_with (str):

            Returns: a set of tuples (Tag, key) that the KEY inputted is coupled with

        """
        # list of strings of the `Tag:Key` or `Key` listed in 'Coupled'
        coupled_with = [x.strip() for x in unparsed_coupled_with.split(',')]
        # check if each item in list reports its Tag
        # (looking for strings that are of the form `Tag:Key`, else assume given TAG)
        coupled_with_tag = [tag if x.find(':') < 0 else x[:x.find(':')] for x in coupled_with]
        # strip the `Tag:` or `Tag-ID` portion off any strings
        # that follow the form `Tag:Key` or `Tag-ID:Key`
        coupled_with_key = [x if x.find(':') < 0 else x[x.find(':') + 1:] for x in coupled_with]
        # check if each item in list reports its ID
        # (looking for TAG strings that are of the form `Tag-ID`, else assume .....)
        coupled_with_id = [id_str if x == tag else '' if x.find('-') < 0 else x[:x.find('-')] for x
                           in coupled_with_tag]
        # create set of tuples (tag, key, id) of tag-key values that the given key is coupled with
        tag_key_set = set(zip(coupled_with_tag, coupled_with_key, coupled_with_id))
        return tag_key_set

    @classmethod
    def fetch_coupled(cls, coupled_properties, tag_key_id):
        """ Function to determine and fetch in only the relevant and unique coupled sets
            to the class coupled sensitivity variable. It determines the coupled properties
            first and then it is compared to other existing coupled set to determine if there is
            any subset or intersection. Append it to the class coupled sensitivity variable
            under appropriate conditions.

            Args:
                coupled_properties (set): a set that contains coupled_sets of properties from
                    different types or same type.
                tag_key_id (tuple(str): tuple of the tag and key-ID (ID is optional)
        """
        # check to make sure all tag-keys in coupled_properties are valid inputs in the SCHEMA_TREE
        for coup_tag, coup_key, id_str in coupled_properties:
            schema_key = cls.schema_dct.get("tags").get(coup_tag).get("keys").get(coup_key)
            if schema_key is None:
                cls.report_warning('cannot couple', tag_key=tag_key_id,
                                   schema_key=(coup_tag, coup_key, id_str))
        # the coupled_properties should not already have the property name, then add it to the
        # coupled_properties because they are coupled
        coupled_properties.add(tag_key_id)

        for coupled_set in cls.sensitivity['coupled']:
            # checking if coupled_properties is subset for any coupled_set that already
            # existed in the class sensitivity; no need to add if yes.
            if coupled_properties.issubset(coupled_set):
                return
            # checking if coupled_properties and coupled_set has any intersection;
            # if yes, union them to replace the smaller subsets
            if len(coupled_properties.intersection(coupled_set)):
                cls.sensitivity['coupled'].append(coupled_properties.union(coupled_set))
                cls.sensitivity['coupled'].remove(coupled_set)
                return
        # if class coupled sensitivity is empty or neither of the above conditions holds,
        # just add to sensitivity['coupled']
        cls.sensitivity['coupled'].append(coupled_properties)

    @classmethod
    def bad_active_combo(cls, **kwargs):
        """ Based on what the user has indicated as active (and what the user has not), predict
        whether or not the simulation(s) will have trouble solving.

        Returns (bool): True if there are errors found. False if there are no errors found in the
        errors log.

        """
        dervet = kwargs.get('dervet', False)
        other_ders_active = kwargs.get('other_ders', False)
        slf = cls.template
        if slf.Scenario is None:
            TellUser.error('Please activate the Scenario tag and re-run.')
            return True

        if slf.Finance is None:
            TellUser.error('Please activate the Finance tag and re-run.')
            return True

        if slf.RA and slf.DR:
            TellUser.error('Please pick either Resource Adequacy or Demand Response. ' +
                           'They are not compatible with each other.')
            return True

        if not len(slf.Battery) and not len(slf.PV) and not len(slf.CAES) \
                and not len(slf.ICE) and slf.Load == None and not other_ders_active:
            TellUser.error('You forgot to select a technology to include in the analysis. ' +
                           'Please define an energy source.')
            return True

        # require +1 energy market participation when participting in any ancillary services
        if (slf.SR or slf.NSR or slf.FR or slf.LF) and not (slf.retailTimeShift or slf.DA):
            TellUser.error('We require energy market participation when participating in ' +
                           'ancillary services (ie SR, NSR, FR, LF). Please activate the DA ' +
                           'service or the RT service and run again.')
            return True

        if not dervet and slf.DA is None and slf.retailTimeShift is None:
            TellUser.error('Not providing DA or retailETS might cause the solver to take ' +
                           'infinite time to solve!')
            return True

        if not dervet and len(slf.CAES) and len(slf.Battery):
            TellUser.error("Storage technology CAES and Battery should not be active " +
                           "together in StorageVET.")
            return True

        return False

    @classmethod
    def read_referenced_data(cls):
        """ This function makes a unique set of filename(s) based on grab_value_lst.
        It applies for time series filename(s), monthly data filename(s), customer tariff
        filename(s), and cycle life filename(s). For each set, the corresponding class dataset
        variable (ts, md, ct, cl) is loaded with the data.

        Preprocess monthly data files

        """
        ts_files = cls.grab_value_set('Scenario', 'time_series_filename')
        md_files = cls.grab_value_set('Scenario', 'monthly_data_filename')
        ct_files = cls.grab_value_set('Finance', 'customer_tariff_filename')
        yr_files = cls.grab_value_set('Finance', 'yearly_data_filename')
        cl_files = cls.grab_value_set('Battery', 'cycle_life_filename')

        for ts_file in ts_files:
            cls.referenced_data['time_series'][ts_file] = \
                cls.read_from_file('time_series', ts_file, 'Datetime (he)')
        for md_file in md_files:
            cls.referenced_data['monthly_data'][md_file] = \
                cls.read_from_file('monthly_data', md_file, ['Year', 'Month'])
        for ct_file in ct_files:
            cls.referenced_data['customer_tariff'][ct_file] = \
                cls.read_from_file('customer_tariff', ct_file, 'Billing Period')
        for yr_file in yr_files:
            cls.referenced_data['yearly_data'][yr_file] = \
                cls.read_from_file('yearly_data', yr_file, 'Year')
        for cl_file in cl_files:
            cls.referenced_data['cycle_life'][cl_file] = \
                cls.read_from_file('cycle_life', cl_file)

        return True

    @classmethod
    def grab_value_set(cls, tag, key):
        """ Checks if the tag-key exists in cls.sensitivity, otherwise grabs the base case value
        from cls.template

        Args:
            tag (str): tag
            key (str): key that that value(s) represent

        Returns: set of values

        """
        temp_lst_values = []
        try:
            id_str_keys = getattr(cls.template, tag).keys()
        except TypeError:
            return set()
        for id_str in id_str_keys:
            try:
                temp_lst_values += list(cls.sensitivity['attributes'][(tag, key, id_str)])
            except KeyError:
                temp_lst_values += [getattr(cls.template, tag)[id_str][key]]
        return set(temp_lst_values)

    @classmethod
    def build_case_definitions(cls):
        """ Method to create a dataframe that includes all the possible combinations of sensitivity
        values with coupling filter.

        """
        sense = cls.sensitivity["attributes"]
        if not len(sense):
            return

        # apply final coupled filter, remove sets that are not of equal length
        # i.e. select sets only of equal length
        cls.sensitivity['coupled'][:] = [x for x in cls.sensitivity['coupled'] if
                                         cls.equal_coupled_lengths(x)]
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
                coupled_dict = {coupled_prop: sensitivity_case[coupled_prop] for coupled_prop in
                                coupled_set}
                # iterate through possible tag-key-value coupled combos
                for ind in range(coupled_set_value_length):
                    # create a dictionary where :
                    # value is the IND-th value of PROP within SENSE and each key is PROP
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

    @classmethod
    def equal_coupled_lengths(cls, coupled_set):
        """ Check class sensitivity variable if there is any coupled set has unequal lengths of
        coupling arrays

            Args:
                coupled_set (set): a coupled_set that contains Tag/KeyID tuple - Property name of
                the sensitivity arrays being coupled

            Returns:
                True if all those arrays have the same length, the returned set has length of 1 to
                contain unique value False if those arrays have different lengths
        """

        prop_lens = set(map(len, [cls.sensitivity['attributes'][x] for x in coupled_set]))
        if len(prop_lens) == 1:
            return True  # only one unique value length in the set means all list equal length
        else:
            no_cases = np.prod(list(prop_lens))
            message = f"coupled sensitivity arrays related to {list(coupled_set)[0]} do not " \
                      "have the same length; the total number of cases generated from this " \
                      f'coupled_set is {no_cases} because coupling ability for this is ignored.'
            TellUser.warning(message)
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

            # replace any backslashes with forward slash
            filename = filename.replace('\\', '/')

            # logic for time_series data
            parse_dates = name == 'time_series'
            infer_dttm = name == 'time_series'

            # select read function based on file type
            func = pd.read_csv if ".csv" == Path(filename).suffix else pd.read_excel

            try:
                raw = func(filename, parse_dates=parse_dates, index_col=ind_col,
                           infer_datetime_format=infer_dttm)
            except UnicodeDecodeError:
                try:
                    raw = func(filename, parse_dates=parse_dates, index_col=ind_col,
                               infer_datetime_format=infer_dttm,
                               encoding="ISO-8859-1")
                except (ValueError, IOError):
                    raise FilenameError(
                        f"Could not open {name} at '{filename}' from: {os.getcwd()}")
                else:
                    TellUser.info(f"Successfully read in: {filename}")
            except (ValueError, IOError):
                raise FilenameError(
                    f"Could not open or could not find {name} at '{filename}' from: {os.getcwd()}")
            else:
                TellUser.info("Successfully read in: " + filename)

        if name == 'customer_tariff':
            # for retail tariff file data, redefine the indexes as strings
            # this ensures a consistent results/simple_monthly_bill.csv file
            # NOTE: the GUI uses unique strings (not integers) for the Billing Period
            raw.index = raw.index.astype(str)

        return raw

    @classmethod
    def case_builder(cls):
        """
            Function to create all the instances based on the possible combinations of sensitivity
            values provided to update the variable instances of this Params class object

        """
        dictionary = {}
        case = copy.deepcopy(cls.template)
        # while case definitions is not an empty df (there is SA) or if it is the last row in
        # case definitions
        if not cls.case_definitions.empty:
            for index in cls.case_definitions.index:
                row = cls.case_definitions.iloc[index]
                for col in row.index:
                    case.modify_attribute(tup=col, value=row[col])
                dictionary.update({index: case})
                case = copy.deepcopy(cls.template)
        else:
            dictionary.update({0: case})

        cls.instances = dictionary

    def modify_attribute(self, tup, value):
        """ Function to modify the value of the sensitivity value based on the specific
        instance/scenario

            Args:
                tup (tuple): the group of attributes
                value (int/float): the new value to replace the former value

        """

        attribute = getattr(self, tup[0])
        if len(tup) == 3:
            attribute = attribute[tup[2]]
        attribute[tup[1]] = value

    @classmethod
    def load_and_prepare(cls):
        """ Flattens each tag that the Schema has defined to only have 1 allowed.

        Returns a params class where the tag attributes that are not allowed to have more than one
        set of key inputs are just dictionaries of their key inputs (while the rest remain
        dictionaries of the sets of key inputs)
        """
        tag_tree = cls.schema_dct.get("tags")
        for case, slf in cls.instances.items():
            TellUser.info(f"Loading case {case}...")
            for tag, tag_attr in tag_tree.items():
                if tag not in ['Results']:
                    max_num = tag_attr.get('max_num')
                    if max_num is not None and int(max_num) == 1:
                        setattr(slf, tag, slf.flatten_tag_id(getattr(slf, tag)))
            slf.load_data_sets()
            slf.load_scenario()
            slf.load_finance()
            slf.load_technology()
            slf.load_services()
        # report back any warning
        if cls.input_error_raised:
            #TellUser.close_log()
            raise ModelParameterError(
                "The model parameter has some errors associated to it. Please fix and rerun.")
        if cls.timeseries_missing_error:
            raise TimeseriesMissingError(
                "A required column of data in the input time series CSV is missing. " +
                "Check the log file, fix and rerun.")
        if cls.timeseries_data_error:
            raise TimeseriesDataError(
                "The values of the time series data has some errors associated" +
                " to it. Check the log file, fix and rerun.")
        if cls.monthly_error:
            raise MonthlyDataError(
                "The values of the monthly data has some errors associated" +
                " to it. Check the log file, fix and rerun.")

    @staticmethod
    def flatten_tag_id(tag_id_dictionary):
        """flatten attributes who should only have 1 subelement – save the value as the attribute
            (there should only be one item in the TAG_ID_DICTIONARY)

        Args:
            tag_id_dictionary (dict): result of the read_and_validate method

        Returns (Dict): value of TAG_ID_DICTIONARY

        """
        if tag_id_dictionary is None or not len(tag_id_dictionary):
            return None
        return dict(*tag_id_dictionary.values())

    def load_data_sets(self):
        """Assign time series data, cycle life data, and monthly data to Scenario dictionary based
        on the dataset content type. Assign customer tariff data to Finance dictionary based on
        the customer tariff filename. These successful assignments is recorded in the developer
        log.
        """
        scenario = self.Scenario
        finance = self.Finance

        self.set_copy_of_referenced_data("time_series", scenario)
        self.set_copy_of_referenced_data("monthly_data", scenario)
        self.set_copy_of_referenced_data("customer_tariff", finance)
        self.set_copy_of_referenced_data("yearly_data", finance)
        if self.Battery is not None:
            # iterate through sets of Battery inputs
            for battery_input_tree in self.Battery.values():
                self.set_copy_of_referenced_data("cycle_life", battery_input_tree)

        TellUser.info("Data sets are loaded successfully.")

    def set_copy_of_referenced_data(self, filename, tag_tree):
        """ Grabs referenced data and adds it to the TAG_TREE dictionary with the key FILENAME
            Args:
                filename (str): name of the file to be added (FILENAME + "_filename")
                tag_tree (dict): tag to save data in

        """
        filepath = tag_tree[f"{filename}_filename"]
        tag_tree[filename] = copy.deepcopy(self.referenced_data[filename][filepath])

    def load_scenario(self):
        """ Error checks, interprets user given data and prepares it for Scenario initialization.

        """
        scenario = self.Scenario
        raw_time_series = scenario["time_series"]
        raw_monthly_data = scenario['monthly_data']
        dt = scenario['dt']
        opt_years = scenario['opt_years']

        # find frequency (indicated by user's dt input)
        freq, dt_exact = self.stringify_delta_time(dt)
        # make sure freq is not an empty string, otherwise report and raise error immediately
        if not freq:
            self.record_input_error(
                f'The timestep frequency cannot be determined from the input value of dt ({dt}). ' +
                'Please use a number representing a regular fraction of an hour. ' +
                'Options are 1 (1 hour), ' +
                '0.5 (30 minutes), 0.25 (15 minutes), 0.166 (10 minutes), ' +
                '0.083 (5 minutes), or 0.016 (1 minute).')
            raise ModelParameterError(
                "The model parameter has some errors associated to it. Please fix and rerun.")
        else:
            TellUser.info(f'The timestep frequency was set to "{freq}" based on dt = {dt} hours')
        scenario['frequency'] = freq
        # set dt to be exactly dt
        scenario['dt'] = dt_exact
        # process time series and save it in place of the raw data
        time_series = self.process_time_series(raw_time_series, freq, dt_exact, opt_years)
        scenario["time_series"] = time_series
        # report on any missing values from each time series
        scenario["time_series_nan_count"] = self.count_nans_time_series(time_series)
        # process monthly data and save it in place of the raw data
        monthly_data = self.process_monthly(raw_monthly_data, opt_years)
        scenario["monthly_data"] = monthly_data

        # make sure that the project start year is before the project end year
        # ignore this check if dervet's analysis_horizon_mode is not set to 1
        # (in those cases, end_year is redefined downstream based on expected lifetimes)
        if self.Finance.get('analysis_horizon_mode', 1) == 1:
            if scenario['end_year'].year < scenario['start_year'].year:
                self.record_input_error(
                    f"end_year ({scenario['end_year'].year}) < start_year " +
                    f"({scenario['start_year'].year}). end_year should be later than start_year.")
        # determine if mpc
        # self.Scenario['mpc'] = (self.Scenario['n_control'] != 0)
        scenario['mpc'] = False

        # load POI inputs
        self.POI = {
            'max_export': scenario['max_export'],
            'max_import': scenario['max_import'],
            'load_growth': scenario['def_growth'],
            'apply_poi_constraints': scenario['apply_interconnection_constraints'],
        }
        TellUser.info("Successfully prepared the Scenario")

    @staticmethod
    def stringify_delta_time(dt):
        """ Based on the value of the input parameter dt, return the frequency that the pandas data time index
        should take, along with the exact decimal floating point value

        Args:
            dt (float): user reported time-step of time series data frame

        Returns: a string representation of the frequency of the time series index,
                 and the exact floating point representation of dt

        """
        frequency = ''
        dt_exact = 0.0
        dt_truncated = truncate_float(dt)
        if dt_truncated == 1.0:
            frequency = 'H'
            dt_exact = 60 / 60
        if dt_truncated == 0.5:
            frequency = '30min'
            dt_exact = 30 / 60
        if dt_truncated == 0.25:
            frequency = '15min'
            dt_exact = 15 / 60
        if dt_truncated == 0.166 or dt_truncated == 0.167:
            frequency = '10min'
            dt_exact = 10 / 60
        if dt_truncated == 0.083:
            frequency = '5min'
            dt_exact = 5 / 60
        if dt_truncated == 0.016 or dt_truncated == 0.017:
            frequency = '1min'
            dt_exact = 1 / 60
        return frequency, dt_exact

    def get_single_series(self, ts, column_name, nan_count, description=None, bypass_key_error=False, allow_nans=False):
        # if the column name does not exist in the data, an error ensues
        # if NaNs are found in the data, the nan_count is recorded and an error ensues
        #   return None on any error
        # description is an optional argument
        if description is None:
            description = f"'{column_name}'"
        single_ts = None
        try:
            single_ts = ts.loc[:, column_name]
            nan_count = nan_count[column_name]
            if nan_count != 0:
                if allow_nans:
                    TellUser.warning(f"The input timeseries data: '{column_name}' " +
                                     f"has {nan_count} missing/NaN value(s). These will become " +
                                      "filled in accordingly.")
                else:
                    self.record_timeseries_data_error(
                        f"The input timeseries data: '{column_name}' has {nan_count} " +
                        "missing/NaN value(s). Please make sure that each value is a number.")
        except KeyError:
            if bypass_key_error:
                pass
            else:
                self.record_timeseries_missing_error(
                    f"Missing '{column_name}' from timeseries input. " +
                    f"Please include {description} in timeseries csv")
        return single_ts

    def count_nans_time_series(self, time_series):
        # report on the number of nans in each time series
        # returns a dict with an integer value for each time series key (name)
        nans_count = dict()
        for ts_key, ts_val in time_series.items():
            nans_count[ts_key] = ts_val.isnull().sum()
        return nans_count

    def process_time_series(self, time_series, freq, dt, opt_years):
        """ Given a time series data frame, transforms index into time-step beginning and preforms
        basic checks on the data with the scenario defined

        Args:
            time_series (DataFrame): raw time series data frame.
            freq (str): frequency that the date time index should take
            dt (exact floating point value)
            opt_years

        Returns: a DataFrame with the index beginning at hour 0, if an there is an error with the
            raw data that needs to be addressed, then return None

        """
        first_hour = time_series.index.hour[0]
        first_min = time_series.index.minute[0]
        # require that the index begins with hour 1 or hour 0
        if first_hour == 1 or (first_hour == 0 and first_min != 0):
            time_series = time_series.sort_index()  # make sure all the time_stamps are in order
            yr_included = time_series.index.shift(-1, freq).year.unique()
        elif first_hour == 0 and first_min == 0:
            time_series = time_series.sort_index()  # make sure all the time_stamps are in order
            yr_included = time_series.index.year.unique()
        else:
            self.record_timeseries_data_error(
                'The time series does not start at the beginning of the day. ' +
                'Please start with hour as 1 or 0.')
            return time_series

        # check that leap years are taken into account, else tell the user
        hours_in_years = [8784 if is_leap_yr(data_year) else 8760 for data_year in yr_included]
        expected_data_length = sum(hours_in_years) / dt
        if len(time_series) != expected_data_length:
            self.record_timeseries_data_error(
                f"The expected data length does not match with the length of the given data " +
                f"({len(time_series)}). The expected data length (for {freq} data) is " +
                f"sum({hours_in_years}) / {dt} = {expected_data_length}")
            return time_series

        # replace time_series index with pandas formatted datetime index
        # sets index to denote the start of the time measurement
        new_indx = create_timeseries_index(yr_included, freq)
        time_series.index = new_indx
        time_series.columns = [column.strip() for column in time_series.columns]
        # NOTE: empty data values are left empty

        # make sure that opt_years are defined in time series data
        time_series_index = time_series.index
        if set(opt_years) != set(yr_included):
            self.record_timeseries_data_error(
                f"The 'opt_years' input should coinside with data in the " +
                f"Time Series file. {opt_years} != {set(yr_included)}")
        return time_series

    @classmethod
    def record_timeseries_data_error(cls, error_message):
        """ logs an error message to the user and mark the class as un-able to run.

        Args:
            error_message (str): string error message

        """
        TellUser.error(error_message)
        cls.timeseries_data_error = True

    @classmethod
    def record_timeseries_missing_error(cls, error_message):
        """ logs an error message to the user and mark the class as un-able to run.

        Args:
            error_message (str): string error message

        """
        TellUser.error(error_message)
        cls.timeseries_missing_error = True

    def process_monthly(self, monthly_data, opt_years):
        """ processing monthly data.
        Creates monthly Period index from year and month

        Args:
            monthly_data (DataFrame): Raw df
            opt_years (list)

        Returns:
            monthly_data (DataFrame): edited df

        """
        if not monthly_data.empty:
            monthly_data.index = pd.PeriodIndex(
                year=monthly_data.index.get_level_values(0).values,
                month=monthly_data.index.get_level_values(1).values,
                freq='M')
            monthly_data.index = monthly_data.index.to_timestamp()
            monthly_data.index.name = 'yr_mo'
            monthly_data.columns = [column.strip() for column in monthly_data.columns]
            yr_included = monthly_data.index.year.unique()
            if set(opt_years) != set(yr_included):
                self.record_monthly_error(
                    f"The 'opt_years' input should coinside with data in the " +
                    f"Monthly file. {opt_years} != {set(yr_included)}")
        return monthly_data

    @classmethod
    def record_monthly_error(cls, error_message):
        """ logs an error message to the user and mark the class as un-able to run.

        Args:
            error_message (str): string error message

        """
        TellUser.error(error_message)
        cls.monthly_error = True

    @classmethod
    def record_input_error(cls, error_message):
        """ logs an error message to the user and mark the class as un-able to run.

        Args:
            error_message (str): string error message

        """
        TellUser.error(error_message)
        cls.input_error_raised = True

    def load_finance(self):
        """ Interprets user given data and prepares it for Finance.

        """
        # include data in financial class
        self.Finance.update({
            'n': self.Scenario['n'],
            'mpc': self.Scenario['mpc'],
            'start_year': self.Scenario['start_year'],
            'end_year': self.Scenario['end_year'],
            'CAES': self.CAES is not None,
            'dt': self.Scenario['dt'],
            'opt_years': self.Scenario['opt_years'],
            'def_growth': self.Scenario['def_growth'],
            'frequency': self.Scenario['frequency'],
            'verbose': self.Scenario['verbose'],
            'customer_sided': self.DCM is not None or self.retailTimeShift is not None
        })

        TellUser.info("Successfully prepared the Finance")

    def load_technology(self, names_list=None):
        """ Error checks, interprets DER user given data and prepares for
        initialization.

        """
        scenario = self.Scenario
        time_series = scenario['time_series']
        time_series_nan_count = scenario['time_series_nan_count']
        monthly_data = scenario['monthly_data']
        freq = scenario['frequency']
        dt = scenario['dt']
        binary = scenario['binary']

        if names_list is None:
            # then no name_lst was inherited so initialize as list type
            names_list = []

        # validation checks for a CAES's parameters
        for id_str, caes_inputs in self.CAES.items():
            names_list.append(caes_inputs['name'])
            if caes_inputs['ch_min_rated'] > caes_inputs['ch_max_rated']:
                self.record_input_error(f"CAES #{id_str} ch_max_rated < " +
                                        f"ch_min_rated. ch_max_rated should " +
                                        f"be greater than ch_min_rated")

            if caes_inputs['dis_min_rated'] > caes_inputs['dis_max_rated']:
                self.record_input_error(f"CAES #{id_str} dis_max_rated < " +
                                        f"dis_min_rated. dis_max_rated " +
                                        f"should be greater than " +
                                        f"dis_min_rated")
            # add scenario case parameters to CAES dictionary
            caes_inputs.update({'binary': binary, 'dt': dt})

        # validation checks for a Battery's parameters
        for id_str, bat_input in self.Battery.items():
            # max ratings should be greater than min rating - power and energy
            if bat_input['ch_min_rated'] > bat_input['ch_max_rated']:
                self.record_input_error(f"Battery #{id_str} ch_max_rated < " +
                                        f"ch_min_rated. ch_max_rated should " +
                                        f"be greater than ch_min_rated")

            if bat_input['dis_min_rated'] > bat_input['dis_max_rated']:
                self.record_input_error(
                    f"Battery #{id_str} dis_max_rated < dis_min_rated. dis_max_rated should be greater than dis_min_rated")

            if bat_input['ulsoc'] < bat_input['llsoc']:
                self.record_input_error(
                    f"Battery #{id_str} ulsoc < llsoc. ulsoc should be greater than llsoc")

            if bat_input['soc_target'] < bat_input['llsoc']:
                self.record_input_error(
                    f"Battery #{id_str} soc_target < llsoc. soc_target should be greater than llsoc")

            # add scenario case parameters to battery parameter dictionary
            bat_input.update({'binary': binary,
                              'dt': dt})
            names_list.append(bat_input['name'])

        # add scenario case parameters to PV parameter dictionary
        for id_str, pv_input_tree in self.PV.items():
            column_name = 'PV Gen (kW/rated kW)'
            pv_input_tree['dt'] = dt
            names_list.append(pv_input_tree['name'])
            # first attempt to load the time series without the id_str, if there is a single instance of this technology
            if len(self.PV.keys()) == 1:
                ts_rated_gen = self.get_single_series(time_series, column_name, time_series_nan_count, 'PV Generation', bypass_key_error=True)
                if ts_rated_gen is not None:
                    pv_input_tree.update({'rated gen': ts_rated_gen})
                    continue
            # second attempt to load the time series with the id_str
            pv_input_tree.update({'rated gen': self.get_single_series(time_series, f'{column_name}/{id_str}', time_series_nan_count, 'PV Generation')})

        # add scenario case parameters to ICE parameter dictionary
        for ice_input in self.ICE.values():
            ice_input['dt'] = dt
            names_list.append(ice_input['name'])

        if len(set(names_list)) != len(names_list):
            self.record_input_error('The input names for one or more of the ' +
                                    'DER technologies (Battery, CAES, ICE, ' +
                                    'PV, etc) are not unique.')

        if scenario['incl_site_load']:
            # Load is not included in MP sheet (so no scalar inputs)
            # If the user wants to include load in analysis,
            # it is indicated by the 'INCL_SITE_LOAD' flag
            key = 'incl_site_load'
            self.Load = {key: {'name': "Site Load",
                               'power_rating': 0,
                               'duration': 0,
                               'startup_time': 0,
                               'dt': dt,
                               'construction_year': scenario['start_year'] - 1,
                               'operation_year': scenario['start_year'],
                               'nsr_response_time': 0,
                               'sr_response_time': 0,
                               'decommissioning_cost': 0,
                               'salvage_value': 0,
                               'expected_lifetime': int(1e3),
                               'replaceable': 0,
                               'acr': 0,
                               'ter': 0,
                               'ecc%': 0,
                               }
                         }
            self.Load[key]['site_load'] = self.get_single_series(time_series, 'Site Load (kW)', time_series_nan_count, 'Site Load')

        TellUser.info("Successfully prepared the Technologies")

    @staticmethod
    def monthly_to_timeseries(freq, column):
        """ Converts data given monthly into timeseries.

        Args:
            freq (str): frequency that the date time index should take
            column (DataFrame):  A column of monthly data given by the user,
                indexed by month

        Returns: A Series of timeseries data that is equivalent to the column
            given

        """
        first_day = f"1/1/{column.index.year[0]}"
        last_day = f"1/1/{column.index.year[-1] + 1}"

        new_index = pd.date_range(start=first_day, end=last_day, freq=freq,
                                  closed='left')
        temp = pd.DataFrame(index=new_index)
        temp['yr_mo'] = temp.index.to_period('M')
        column['yr_mo'] = column.index.to_period('M')
        temp = temp.reset_index().merge(column.reset_index(drop=True), on='yr_mo',
                                        how='left').set_index(new_index)
        return temp[column.columns[0]]

    def load_services(self):
        """ Error checks, interprets Value Stream user given data and prepares
        it for initialization.

        """
        scenario = self.Scenario  # dictionary of scenario inputs
        monthly_data = scenario['monthly_data']
        time_series = scenario['time_series']
        time_series_nan_count = scenario['time_series_nan_count']
        dt = scenario['dt']
        freq = scenario['frequency']

        if self.Deferral is not None:
            self.Deferral.update({
                'load': self.get_single_series(time_series, 'Deferral Load (kW)', time_series_nan_count, 'Deferral Load'),
                'last_year': scenario["end_year"],
                'dt': dt
            })

        if self.Volt is not None:
            self.Volt['dt'] = dt
            percent = self.get_single_series(time_series, 'VAR Reservation (%)', time_series_nan_count,
                      'Vars Reservations as a percent (0 to 100) of inverter max')
            # make sure that vars reservation inputs are between 0 and 100
            if len([*filter(lambda x: x > 100, percent.value)]) or len(
                    [*filter(lambda x: x < 0, percent.value)]):
                self.record_timeseries_data_error(
                    "Value error within 'Vars Reservation (%)' timeseries. Please " +
                    "include the vars reservation as percent (from 0 to 100) of inverter max.")
            self.Volt['percent'] = percent

        if self.RA is not None:
            if 'active hours' in self.RA['idmode'].lower():
                # if using active hours, require the column
                self.RA['active'] = self.get_single_series(time_series, 'RA Active (y/n)', time_series_nan_count, 'when RA is active')
            try:
                self.RA['value'] = monthly_data.loc[:, 'RA Capacity Price ($/kW)']
            except KeyError:
                self.record_input_error(
                    "Missing 'RA Price ($/kW)' from monthly CSV input. Please include monthly RA price.")
            self.RA['system_load'] = self.get_single_series(time_series, "System Load (kW)", time_series_nan_count, 'System Load')
            self.RA.update({'default_growth': scenario['def_growth'],
                            # applied to system load if 'forecasting' load
                            'dt': dt})

        if self.DR is not None:
            try:
                self.DR['cap_price'] = monthly_data.loc[:, 'DR Capacity Price ($/kW)']
            except KeyError:
                self.record_monthly_error(
                    "Missing 'DR Capacity Price ($/kW)' from monthly input. lease include DR capcity prices.")
            try:
                self.DR['ene_price'] = monthly_data.loc[:, 'DR Energy Price ($/kWh)']
            except KeyError:
                self.record_monthly_error(
                    "Missing 'DR Energy Price ($/kWh)' from monthly input. Please include DR energy prices")
            try:
                self.DR['dr_months'] = self.monthly_to_timeseries(freq, monthly_data.loc[:,['DR Months (y/n)']])
            except KeyError:
                self.record_monthly_error(
                    "Missing 'DR Months (y/n)' from monthly input. Please include DR months.")
            try:
                self.DR['dr_cap'] = self.monthly_to_timeseries(freq, monthly_data.loc[:,['DR Capacity (kW)']])
                self.DR['cap_monthly'] = monthly_data.loc[:, "DR Capacity (kW)"]
            except KeyError:
                self.record_monthly_error(
                    "Missing 'DR Capacity (kW)' from monthly input. Please include a DR capacity.")
            self.DR['system_load'] = self.get_single_series(time_series, "System Load (kW)", time_series_nan_count, 'System Load')
            self.DR.update({'default_growth': scenario['def_growth'],
                            'dt': dt})

        if self.Backup is not None:
            self.Backup['dt'] = dt
            try:
                self.Backup['monthly_price'] = monthly_data.loc[:, 'Backup Price ($/kWh)']
            except KeyError:
                self.record_input_error(
                    "Missing 'Backup Price ($/kWh)' from monthly data input. Please include Backup Price in monthly data csv.")

            try:
                self.Backup.update({'daily_energy': self.monthly_to_timeseries(freq,
                                                                               monthly_data.loc[:,
                                                                               [
                                                                                   'Backup Energy (kWh)']]),
                                    'monthly_energy': monthly_data.loc[:, 'Backup Energy (kWh)']})
            except KeyError:
                self.record_input_error(
                    "Missing 'Backup Energy (kWh)' from monthly data input. Please include Backup Energy in monthly data csv.")
        if self.SR is not None:
            self.SR['dt'] = dt
            self.SR.update({'price': self.get_single_series(time_series, 'SR Price ($/kW)', time_series_nan_count, 'Spinning Reserve Price')})

        if self.NSR is not None:
            self.NSR['dt'] = dt
            self.NSR.update({'price': self.get_single_series(time_series, 'NSR Price ($/kW)', time_series_nan_count, 'Non-Spinning Reserve Price')})

        if self.DA is not None:
            self.DA['dt'] = dt
            self.DA.update({'price': self.get_single_series(time_series, 'DA Price ($/kWh)', time_series_nan_count, 'DA ETS price')})

        if self.FR is not None:
            self.FR['dt'] = dt
            self.FR.update({'energy_price': self.get_single_series(time_series, 'DA Price ($/kWh)', time_series_nan_count, 'DA ETS price')})
            if self.FR['CombinedMarket']:
                fr_price_ts = self.get_single_series(time_series, 'FR Price ($/kW)', time_series_nan_count, 'Frequency Regulation price')
                if fr_price_ts is not None:
                    self.FR.update(
                        {'regu_price': np.divide(fr_price_ts, 2),
                         'regd_price': np.divide(fr_price_ts, 2)})
            else:
                self.FR.update(
                    {'regu_price': self.get_single_series(time_series, 'Reg Up Price ($/kW)', time_series_nan_count, 'Frequency Regulation Up price'),
                     'regd_price': self.get_single_series(time_series, 'Reg Down Price ($/kW)', time_series_nan_count, 'Frequency Regulation Down price')})

        if self.LF is not None:
            self.LF['dt'] = dt
            self.LF.update({'energy_price': self.get_single_series(time_series, 'DA Price ($/kWh)', time_series_nan_count, 'DA ETS price')})
            if self.LF['CombinedMarket']:
                lf_price_ts = self.get_single_series(time_series, 'LF Price ($/kW)', time_series_nan_count, 'Load Following price')
                if lf_price_ts is not None:
                    self.LF.update(
                        {'regu_price': np.divide(lf_price_ts, 2),
                         'regd_price': np.divide(lf_price_ts, 2)})
            else:
                self.LF.update(
                    {'regu_price': self.get_single_series(time_series, 'LF Up Price ($/kW)', time_series_nan_count, 'Load Following Up price'),
                     'regd_price': self.get_single_series(time_series, 'LF Down Price ($/kW)', time_series_nan_count, 'Load Following Down price')})
            self.LF['eou'] = self.get_single_series(time_series, 'LF Energy Option Up (kWh/kW-hr)', time_series_nan_count, 'Load Following offset')
            self.LF['eod'] = self.get_single_series(time_series, 'LF Energy Option Down (kWh/kW-hr)', time_series_nan_count, 'Load Following offset')

        if self.User is not None:
            # check to make sure the user included at least one of the custom constraints
            input_cols = ["POI: Max Export (kW)", "POI: Min Export (kW)",
                          "POI: Max Import (kW)", "POI: Min Import (kW)",
                          "Aggregate Energy Max (kWh)", "Aggregate Energy Min (kWh)"]
            if not time_series.columns.isin(input_cols).any():
                self.record_timeseries_missing_error(
                    f"User has not included any of the following time series: {input_cols}. " +
                    "Please add at least one and run again.")
            # power time series columns (first 4 in input_cols list)
            power = pd.DataFrame(index=time_series.index)
            for power_ts_name in np.intersect1d(input_cols[:-2], time_series.columns):
                power = pd.concat([power, self.get_single_series(time_series, power_ts_name, time_series_nan_count)], axis=1)
            # energy time series columns (last 2 in input_cols list)
            energy = pd.DataFrame(index=time_series.index)
            for energy_ts_name in np.intersect1d(input_cols[-2:], time_series.columns):
                energy = pd.concat([energy, self.get_single_series(time_series, energy_ts_name, time_series_nan_count)], axis=1)
            # further restrict values in these time series
            for col in power.columns:
                self.req_no_zero_crossings(power[col].values, col)
            for col in energy.columns:
                self.req_all_non_negative(energy[col].values, col)
            self.User.update({'power': power,
                              'energy': energy,
                              'dt': dt})

        if self.DCM is not None or self.retailTimeShift is not None:
            if scenario['incl_site_load'] != 1:
                self.record_input_error(
                    'Error: you have DCM/retail ETS on, so incl_site_load should be 1. Please fix and re-run')
            tariff = self.Finance['customer_tariff']
            retail_prices = Financial.calc_retail_energy_price(tariff, scenario['frequency'],
                                                               min(scenario['opt_years']))
            if self.retailTimeShift is not None:
                self.retailTimeShift.update({'price': retail_prices.loc[:, 'p_energy'],
                                             'tariff': tariff.loc[tariff.Charge.apply(
                                                 (lambda x: x.lower())) == 'energy', :],
                                             'dt': dt})
            if self.DCM is not None:
                self.DCM.update({'tariff': tariff.loc[
                                           tariff.Charge.apply((lambda x: x.lower())) == 'demand',
                                           :],
                                 'billing_period': retail_prices.loc[:, 'billing_period'],
                                 'dt': dt})

        TellUser.info("Successfully prepared the value-stream (services)")

    @classmethod
    def req_all_non_negative(cls, array, name):
        """ Records a timeseries error is the given array contains ANY negative values

        Args:
            array:
            name:

        """
        if np.any(array < 0):
            cls.timeseries_data_error = True
            TellUser.error(
                f"The input timeseries data: '{name}' contains some negative numbers. "
                + "Please only give positive or zero values.")

    @classmethod
    def req_no_zero_crossings(cls, array, name):
        """ Records a timeseries error if the given array contains positive AND negative values

        Args:
            array:
            name

        """
        if np.any(array < 0) and np.any(array > 0):
            cls.timeseries_data_error = True
            TellUser.error(
                f"The input timeseries data: '{name}' contains some negative and some positive values. "
                + "This column constrains power in one direction. "
                + "Please choose either all positive values or all negative. Zero values are also allowed.")
