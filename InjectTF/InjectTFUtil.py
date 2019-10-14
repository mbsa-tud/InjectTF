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
import struct
from codecs import decode

import tensorflow as tf


def freeze_graph(
    model_name, path_to_model_files, output_node_names, output_dir, mappings=None
):
    """ Freezes a TensorFlow graph.

    Args:
        model_name (str): name of the model, e.g. `MyModel`.
        path_to_model_files (str): Path to the model folder containing
            checkpoint and meta files.
        output_node_names list(str): List of output node names, e.g. ['output'].
        output_dir (str): The output directory.
        mappings (dict, optional): Dictionary containing input and output mapping.
            Defaults to None.
            Format is:
             {
                 'inputs': [my_input_tensor_name_1, my_input_tensor_name_2],
                 'outputs': [my_output_tensor_name]
             }
    """

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path_to_model_files + model_name + ".meta")

        saver.restore(sess, tf.train.latest_checkpoint(path_to_model_files))

        # TODO: check mappings: is the correct collection created?
        # if mappings were provided, add collections with those items
        # to the graph
        if mappings:
            for key in mappings:
                for item in mappings[key]:
                    tf.add_to_collection(
                        key, sess.graph.get_tensor_by_name(item + ":0")
                    )

        graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=output_node_names,
        )

    os.makedirs(output_dir, exist_ok=True)

    with tf.gfile.GFile(output_dir + model_name + ".pb", "wb") as f:
        f.write(graph_def.SerializeToString())


def load_frozen_graph(path_to_pb_file, file_name, prefix=""):
    """Loads a frozen graph.

    Args:
        path_to_pb_file (str): Path to the frozen model.
        file_name (str): File name of the frozen model.
        prefix (str, optional): Prefix for each node in the graph. Defaults to ''

    Returns:
        The loaded TensorFlow graph (`tf.Graph`).
    """

    # check file extension
    file_name = file_name if (file_name.endswith(".pb")) else file_name + ".pb"

    with tf.io.gfile.GFile(path_to_pb_file + file_name, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        # fix nodes
        # for more information visit:
        # https://www.bountysource.com/issues/36614355-unable-to-import-frozen-graph-with-batchnorm
        for node in graph_def.node:
            if node.op == "RefSwitch":
                node.op = "Switch"
                for index in range(len(node.input)):
                    if "moving_" in node.input[index]:
                        node.input[index] = node.input[index] + "/read"
            elif node.op == "AssignSub":
                node.op = "Sub"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]

    with tf.Graph().as_default() as g:

        tf.import_graph_def(graph_def, name=prefix)

    return g


def get_graph_statistics(graph):
    """Generate a list with graph statistics.

    This includes: number of operations and number of weights

    Args:
        graph (`tf.Graph`): A TensorFlow graph.

    Returns:
        Dictionary containing statistics about the graph.
    """

    res = {}
    res["Operations"] = {}
    op_counter = 0

    for op in graph.get_operations():

        op_counter += 1

        if "Operations" not in res or op.type not in res["Operations"]:
            res["Operations"][str(op.type)] = 1

        else:
            res["Operations"][str(op.type)] += 1

    res["Total operations"] = op_counter

    # gather all const values in the graph
    const_vars = []
    for node in graph.get_operations():
        if node.type == "Const":
            const_vars.append(node)

    total_weights = 0

    for item in const_vars:
        dims = item.get_attr("value").tensor_shape.dim
        current_elem_weights = 1
        for dim in dims:
            current_elem_weights *= dim.size
        total_weights += current_elem_weights

    res["Total weights"] = total_weights

    return res


def float_to_bin32(value):
    """Convert float to 32-bit binary string."""
    [d] = struct.unpack(">L", struct.pack(">f", value))
    return "{:032b}".format(d)


def bin_to_float32(b):
    """Convert binary string to a float."""
    bf = int_to_bytes(int(b, 2), 4)  # 4 bytes needed for IEEE 754 binary32.
    return struct.unpack(">f", bf)[0]


def float_to_bin64(value):
    """Convert float to 64-bit binary string."""
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return "{:064b}".format(d)


def bin_to_float64(b):
    """Convert binary string to a float."""
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack(">d", bf)[0]


# TODO refactor: use int.to_bytes() function instead
def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode("%%0%dx" % (length << 1) % n, "hex")[-length:]
