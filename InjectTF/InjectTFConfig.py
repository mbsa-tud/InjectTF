#
#   TU-Dresden, Institute of Automation (IfA)
#   Student research thesis
#
#   Evaluation of the effects of common Hardware faults
#   on the accuracy of safety-critical AI components
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import yaml


class InjectTFConfig:
    def __init__(self, path_to_config_file="./config/InjectTF_conf.yml", DEBUG=False):
        """InjectTF configuration class.

        This class handles reading and parsing the yaml configuration file and
        provides some utility methods for easier interaction with the parsed data.

        Args:
            path_to_config_file (str): Path to the configuration file. Defaults to
                "./config/InjectTF_conf.yml".
            DEBUG (bool):   Prints debugging information if true. This will print a lot
                to stdout! It is recommended to pipe the output to a log-file.
                Defaults to false.
        """

        self.__DEBUG = DEBUG
        self.__config_data = self.__read_config(path_to_config_file)

    def __read_config(self, file):

        if self.__DEBUG:
            print("Reading config_file: ", file)

        file = (
            file if (file.endswith(".yml") or file.endswith(".yaml")) else file + ".yml"
        )

        try:
            with open(file, "rb") as f:
                data = yaml.safe_load(f)
        except IOError as error:
            print("Can not open file: ", file)
            raise

        if self.__DEBUG:
            print("Done reading config file.")

        return data

    def data(self):
        """Returns a dictionary containing the complete data of the configuration file."""
        return self.__config_data

    def get_seed_for_rand_num_gen(self):
        """Returns the random number generator seed. If no seed has been defined
            in the configuration file, `None` is returned instead."""

        if "Seed" in self.__config_data:
            return self.__config_data["Seed"]

        return None

    def get_scalar_fault_type(self):
        """Returns the scalar fault type."""
        return self.__config_data["ScalarFaultType"]

    def get_tensor_fault_type(self):
        """Returns the tensor fault type."""
        return self.__config_data["TensorFaultType"]

    def get_selected_operations(self):
        """Returns a list with all operations selected for injection"""
        res = []
        for key in self.__config_data["Ops"]:
            res.append(key)
        return res

    def get_fault_probabilities(self):
        """Returns a dictionary of all for fault injection selected operations with
            their corresponding probabilities for injection."""
        return self.__config_data["Ops"]

    def get_probability_for_op(self, op):
        """Returns the pobability for injection for the specified operation.

        Args:
            op (str): The TensorFlow name of the operation (eg. "Add", "Mul",
                "Conv2D", ...)
        """
        return self.__config_data["Ops"][op]
