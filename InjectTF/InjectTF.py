#
#   TU-Dresden, Institute of Automation (IfA)
#   Student research thesis
#
#   Evaluation of the effects of common Hardware faults
#   on the accuracy of safety-critical AI components
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import os

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

import InjectTF.InjectedFunctions as ifunc
import InjectTF.InjectTFConfig as itfconf
import InjectTF.InjectionManager as im
import InjectTF.InjectTFUtil as itfutil

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InjectTF:
    def __init__(
        self,
        mappings,
        inject_conf_file,
        model_name,
        path_to_mdl_files,
        mdl_is_frozen=False,
        output_dir="./out/",
        DEBUG=False,
    ):
        """Main class for TensorFlow graph injection.

        Args:
            mappings (dict): Dictionary containing input and output mappings.
                Format is:
                 {
                     'inputs': [my_input_tensor_name_1, my_input_tensor_name_2],
                     'outputs': [my_output_tensor_name]
                 }
            inject_conf_file (str): Path to the configuration file.
            model_name (str): Name of the model to inject.
            path_to_model_files (str): Path to the checkpoint and meta files or
                `.pb` file if the model is already frozen.
            mdl_is_frozen (bool): If true, a `.pb` file will be loaded from the
                `path_to_model_files` directory.
            output_dir (str): Directory for all outputs of this program. E.g. frozen model.
            DEBUG (bool):   Prints debugging information if true. This will print a lot
                to stdout! It is recommended to pipe the output to a log-file.
                Defaults to false.
        """

        self.__DEBUG = DEBUG
        self.__original_mappings = mappings
        self.__model_name = model_name
        self.__path_to_model_files = path_to_mdl_files

        self.__manager = im.InjectionManager(inject_conf_file, DEBUG)

        if mdl_is_frozen:

            self.__original_graph = self.__load_frozen_graph(
                path_to_mdl_files, model_name, ""
            )

        else:

            output_node_names = []
            for item in mappings["outputs"]:
                output_node_names.append(item)

            self.__freeze_graph(
                model_name, path_to_mdl_files, output_node_names, output_dir
            )

            self.__original_graph = self.__load_frozen_graph(output_dir, model_name, "")

        self.__injected_graph = self.__inject_graph(
            self.__original_graph, self.__manager.selected_operations()
        )

    # This function creates a fault injected counterpart to the "original" graph
    # of the neural network. This is done by iterating over all operations within
    # the neural network and adding all operations that are not selected for
    # injection to a new graph, the injected graph `ig`.
    # If during the iteration an operation that is selected for injection is
    # encounterd, an injected counterpart of that operation is generated and
    # added to the injected graph instead. Therefore, after all operations have
    # been added, there exist two graphs during runtime:
    # the original and the injected one.
    def __inject_graph(self, original_graph, ops_to_inject):

        if self.__DEBUG:
            print("Staring injection.")
            print("The following operations will be injected:")

            for item in ops_to_inject:
                print("\t- " + item)

            print("\nBuilding injected graph...\n")

        # Gather operations of original graph
        ops = original_graph.get_operations()

        ig = tf.Graph()

        with ig.as_default():

            # Each op is either added to the `ig` directly, or an fault injected
            # counterpart is generated and then added to `ig`.
            for op in ops:

                if self.__DEBUG:
                    print("\nCurrent operations in injected graph:")
                    print(ig.get_operations())
                    print("------------------------------\n")

                # for placeholders:
                if op.type == "Placeholder":
                    tf.compat.v1.placeholder(
                        op.get_attr("dtype"), op.get_attr("shape"), op.name
                    )

                    # add to collection for easier access
                    ig.add_to_collection(
                        "inputs", ig.get_tensor_by_name(op.name + ":0")
                    )

                else:

                    output_types = []
                    for output in op.outputs:
                        output_types.append(output.dtype)

                    inputs = []
                    for input in op.inputs:
                        inputs.append(ig.get_tensor_by_name(input.name))

                    # create normal operation
                    if op.type not in ops_to_inject:

                        ig.create_op(
                            op.type,
                            inputs,
                            output_types,
                            name=op.name,
                            attrs=op.node_def.attr,
                            op_def=op.op_def,
                        )

                    # create injected operation
                    else:
                        if self.__DEBUG:
                            print("Injecting ", op)

                        self.__manager.inject_function(op, inputs, output_types)

                        if self.__DEBUG:
                            print("Operation injected.\n")

                    # if output node, add to output collection
                    for item in self.__original_mappings["outputs"]:
                        if op.name == item:

                            ig.add_to_collection(
                                "outputs", ig.get_tensor_by_name(op.name + ":0")
                            )

        if self.__DEBUG:
            print("\n\nAll done.\nResulting graph operations:\n")
            print(ig.get_operations())
            print("\n")

        return ig

    def get_original_collections(self):
        """Returns the collections of the original graph."""

        res = {}
        for key in self.__original_mappings:
            res[key] = [
                self.__original_graph.get_tensor_by_name(name + ":0")
                for name in self.__original_mappings[key]
            ]

        return res

    def get_injected_collections(self):
        """Returns the collections of the injected graph."""
        res = {}

        for key in self.__injected_graph.get_all_collection_keys():
            res[key] = self.__injected_graph.get_collection(key)

        return res

    def run_original(self, fetches, feed_dict):
        """Runs operations and evaluates tensors in `fetches` (non-injected graph).

        This is essentially the same `Session.run()` as for normal TensorFlow
        operations. See the official TensorFLow documentation for more details.

        Args:
            fetches: A single graph element, a list of graph elements,
                or a dictionary whose values are graph elements or lists of graph
                elements.
            feed_dict: A dictionary that maps graph elements to values.

        Returns:
            Either a single value if `fetches` is a single graph element, or
            a list of values if `fetches` is a list, or a dictionary with the
            same keys as `fetches` if that is a dictionary (described above).
            Order in which `fetches` operations are evaluated inside the call
            is undefined.
        """

        if self.__DEBUG:
            print("\nRunning original Graph!")

        with tf.Session(graph=self.__original_graph) as sess:

            return sess.run(fetches, feed_dict=feed_dict)

    def run_injected(self, fetches, feed_dict):
        """Runs operations and evaluates tensors in `fetches` (injected graph).

        This is essentially the same `Session.run()` as for normal TensorFlow
        operations. See the official TensorFLow documentation for more details.

        Args:
            fetches: A single graph element, a list of graph elements,
                or a dictionary whose values are graph elements or lists of graph
                elements.
            feed_dict: A dictionary that maps graph elements to values.

        Returns:
            Either a single value if `fetches` is a single graph element, or
            a list of values if `fetches` is a list, or a dictionary with the
            same keys as `fetches` if that is a dictionary (described above).
            Order in which `fetches` operations are evaluated inside the call
            is undefined.
        """

        if self.__DEBUG:
            print("\nRunning injected Graph!")

        with tf.Session(graph=self.__injected_graph) as sess:

            return sess.run(fetches, feed_dict=feed_dict)

    def get_injection_count(self):
        return self.__manager.get_injection_count()

    def __freeze_graph(
        self,
        model_name,
        path_to_model_files,
        output_node_names,
        output_dir,
        mappings=None,
    ):
        # TODO check mappings: is the collection added to the frozen graph correctly?
        itfutil.freeze_graph(
            model_name, path_to_model_files, output_node_names, output_dir, mappings
        )

    def __load_frozen_graph(self, path_to_pb_file, filename, prefix=""):
        return itfutil.load_frozen_graph(path_to_pb_file, filename, prefix)

    def get_unique_operations_in_graph(self):
        """Returns a list of unique operations within the TensorFlow graph."""

        res = []
        for op in self.__original_graph.get_operations():
            if op.type not in res:
                res.append(op.type)
        return res

    def org_graph_stats(self):
        """Returns a dictionary containing stats about the "original" neural network"""
        return itfutil.get_graph_statistics(self.__original_graph)

    def inj_graph_stats(self):
        """Returns a dictionary containing stats about the neural network with injected faults"""
        return itfutil.get_graph_statistics(self.__injected_graph)
