#
#   TU-Dresden, Institute of Automation (IfA)
#   Student research thesis
#
#   Evaluation of the effects of common Hardware faults
#   on the accuracy of safety-critical AI components
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import struct

import numpy as np
import tensorflow as tf

import InjectTF.InjectTFUtil as itfutil


def no_scalar(dtype, val):
    """Dummy injection, returns the input value without modification."""
    return val


def no_tensor(dtype, tensor):
    """Dummy injection, returns the input value without modification."""
    return tensor


def random_scalar(dtype, max=1.0):
    """Return a random value of type dtype from [0, max]"""
    return tf.cast(np.random.random() * max, dtype)


def random_tensor(dtype, tensor):
    """Random replacement of a tensor value with another one"""
    # The tensor.shape is a tuple, while rand needs linear arguments
    # So we need to unpack the tensor.shape tuples as arguments using *
    res = np.random.rand(*tensor.shape)
    return tf.cast(res, dtype)


def bit_flip_scalar(dtype, val):
    """Flips one random bit of the input value `val` and returns the updated value."""
    # convert float according to IEEE 754,
    # cast to integer for xor operation,
    # convert back to float
    if dtype == np.float32:

        # select random bit
        bitNum = np.random.randint(0, 32)
        val_bin = int(itfutil.float_to_bin32(val), 2)
        val_bin ^= 1 << bitNum
        val_bin = bin(val_bin)[2:].zfill(32)
        val = itfutil.bin_to_float32(val_bin)

    elif dtype == np.float64:

        # select random bit
        bitNum = np.random.randint(0, 64)
        val_bin = int(itfutil.float_to_bin64(val), 2)
        val_bin ^= 1 << bitNum
        val_bin = bin(val_bin)[2:].zfill(64)
        val = itfutil.bin_to_float64(val_bin)

    else:
        raise NotImplementedError("Bit flip is not supported for dtype: ", dtype)

    return tf.cast(val, dtype)


def bit_flip_tensor(dtype, tensor):
    """Flips one random bit of a random element of the tensor and returns the updated tensor."""

    # select random element from tensor by generating
    # random indices based on tensor dimensions
    element = []
    for dimension in tensor.shape:
        element.append(np.random.randint(0, dimension))

    def get_element(tens, *e_indices):
        return tens[e_indices]

    def set_element(val, tens, *e_indices):
        tens[e_indices] = val
        return tens

    element_val = get_element(tensor, *element)

    # convert float according to IEEE 754,
    # cast to integer for xor operation,
    # convert back to float
    if dtype == np.float32:

        # select random bit
        bit_num = np.random.randint(0, 32)
        element_val_bin = int(itfutil.float_to_bin32(element_val), 2)
        element_val_bin ^= 1 << bit_num
        element_val_bin = bin(element_val_bin)[2:].zfill(32)
        element_val = itfutil.bin_to_float32(element_val_bin)

    elif dtype == np.float64:

        # select random bit
        bit_num = np.random.randint(0, 64)
        element_val_bin = int(itfutil.float_to_bin64(element_val), 2)
        element_val_bin ^= 1 << bit_num
        element_val_bin = bin(element_val_bin)[2:].zfill(64)
        element_val = itfutil.bin_to_float64(element_val_bin)

    else:
        raise NotImplementedError("Bit flip is not supported for dtype: ", dtype)

    tensor = set_element(element_val, tensor, *element)

    return tf.cast(tensor, dtype)


def zero_scalar(dtype, val):
    """Returns a zero scalar of type dtype."""
    # val is a dummy parameter for compatibility with randomScalar
    return tf.cast(0.0, dtype)


def zero_tensor(dtype, tensor):
    """Returns a tensor of just zeros of the type dtype with the same shape
        as the provided tensor."""
    res = np.zeros(tensor.shape)
    return tf.cast(res, dtype)


# Dictionary containing all implemented fault types.
fault_types = {
    "None": (no_scalar, no_tensor),
    "Rand": (random_scalar, random_tensor),
    "BitFlip": (bit_flip_scalar, bit_flip_tensor),
    "Zero": (zero_scalar, zero_tensor),
}
