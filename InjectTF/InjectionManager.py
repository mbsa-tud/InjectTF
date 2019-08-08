#
#   TU-Dresden, Institute of Automation (IfA)
#   Student research thesis
#
#   Evaluation of the effects of common Hardware faults
#   on the accuracy of safety-critical AI components
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import tensorflow as tf

import InjectTF.InjectTFConfig as itfconf
import InjectTF.InjectedFunctions as ifunc


class InjectionManager:
    def __init__(self, config_file="./config/InjectTF_conf.yml", DEBUG=False):
        """InjectTF fault injection manager class.

        This class handles all fault injection related actions.

        Args:
            config_file (str): Path to the configuration file. Defaults to
                "./config/InjectTF_conf.yml".
            DEBUG (bool):   Prints debugging information if true. This will print a lot
                to stdout! It is recommended to pipe the output to a log-file.
                Defaults to false.
        """

        self.__DEBUG = DEBUG

        if self.__DEBUG:
            print("Initializing InjectionManager...")

        self.__config = itfconf.InjectTFConfig(config_file, DEBUG)

        self.__ifunc = ifunc.InjectedFunctions(
            self.__config.get_scalar_fault_type(),
            self.__config.get_tensor_fault_type(),
            self.__config.get_seed_for_rand_num_gen(),
        )

        self.__check_operations_to_inject(self.__config.data()["Ops"])

        if self.__DEBUG:
            print("Initializing of InjectionManager done.")

    def __check_operations_to_inject(self, ops_to_inject):

        if self.__DEBUG:
            print("Checking if selected operations are supported for injection...")

        not_supported_ops = []

        for key in ops_to_inject:

            if not self.__ifunc.injection_implemented_for_op(key):
                not_supported_ops.append(key)

        if len(not_supported_ops) > 0:

            raise NotImplementedError(
                "Following operations can not be injected: ", not_supported_ops
            )

        if self.__DEBUG:
            print("All selected operations are supported.\n")

    def selected_operations(self):
        """Returns a dictionary containing the for fault injection selected operations."""
        return self.__config.get_selected_operations()

    def get_injection_count(self):
        """Returns the number of injections that have been performed."""
        return self.__ifunc.get_injection_count()

    def inject_function(self, op, inputs, output_types):
        """Create a fault injected operation.

        Args:
            op (`tf.Operation`): The TensorFlow operation that should be injected.
            inputs (lst(`tf.Tensor`)): The list of Tensor objects representing the
                data inputs of this op.
            output_types (lst(DType)): List of DType objects representing the
                output of this op.

        Returns:
            A TensorFlow operation (`tf.Operation`).
        """

        # Get a correpsonding fault injected function for the operation `op`
        injected_function = self.__ifunc.get_injected_function(
            op, self.__config.get_probability_for_op(op.type)
        )

        # Wrap the injected function with `tf.py_function() to create a TensorFlow
        # operation. This has to be done, as "normal" python functions can not
        # be added to the graph directly.
        return tf.py_function(injected_function, inputs, output_types, name=op.name)
