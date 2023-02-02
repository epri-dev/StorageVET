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
from prettytable import PrettyTable
from storagevet.ErrorHandling import *


class Visualization:

    def __init__(self, params_class):
        self.params_class = params_class

    def class_summary(self):
        """
            Function to summarize the Model_Parameters_Template for previsualization

            Return: the summary table
        """
        input_tags = self.params_class.json_tree
        tree = self.params_class.xmlTree
        treeRoot = None
        if tree is not None:
            treeRoot = tree.getroot()
        schema = self.params_class.schema_dct

        TellUser.info("Printing summary table for class Params")
        table = PrettyTable()
        table.field_names = ["Category", "Tag", "ID", "Active?", "Key", "Value", "Type", "Sensitivity?", "Values", "Coupled with?"]
        if input_tags is not None:
            for tag_name, tag_ids in input_tags.items():
                schemaType = self.get_schema_type(schema, tag_name)
                for id_str, tag_id_attrib in tag_ids.items():
                    activeness = tag_id_attrib.get('active')
                    # don't show inactive rows in detail
                    if activeness[0].lower() == "y" or activeness[0] == "1":
                        keys = tag_id_attrib.get('keys')
                        for key_name, key_attrib in keys.items():
                            sensitivity_attrib = key_attrib.get('sensitivity')
                            table.add_row([schemaType, tag_name, id_str, activeness, key_name, key_attrib.get('opt_value'), key_attrib.get('type'),
                                           sensitivity_attrib.get('active'), sensitivity_attrib.get('value'), sensitivity_attrib.get('coupled')])
                    else:
                        table.add_row([schemaType, tag_name, id_str, activeness, '-', '-', '-', '-', '-', '-'])
        else:
            for tag in treeRoot:
                schemaType = self.get_schema_type(schema, tag.tag)
                activeness = tag.get('active')
                id_str = tag.get('id')
                # don't show inactive rows in detail
                if activeness[0].lower() == "y" or activeness[0] == "1":
                    for key in tag:
                        table.add_row([schemaType, tag.tag, id_str, activeness, key.tag, key.find('Value').text, key.find('Type').text,
                                       key.get('analysis'), key.find('Sensitivity_Parameters').text, key.find('Coupled').text])
                else:
                    table.add_row([schemaType, tag.tag, activeness, '-', '-', '-', '-', '-', '-'])

        TellUser.info('User input summary: \n' + str(table))
        return table

    @staticmethod
    def get_schema_type(schema_dict, component_name):
        """ Looks in the schema XML for the type of the component. Used to print the class summary for previsualization.
            Ex: storage-type, storage-technology, pre-dispatch, services, generator, finance, or scenario
        Args:
            schema_dict (dict): the schema
            component_name (str): name of the attribute to be searched and determined for the type

        Returns: the type of the attribute, if found. otherwise it returns "other"

        """
        tag_dicts = schema_dict.get("tags")
        for tag_name, tag_attrib in tag_dicts.items():
            if tag_name == component_name:
                if tag_attrib.get('type') is None:
                    return "other"
                else:
                    return tag_attrib.get('type')
