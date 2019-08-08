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
import numpy as np

import InjectTF.FaultTypes as ftypes


class InjectedFunctions:

    __inject_count = 0

    def __init__(
        self, inj_type_scalar, inj_type_tensor, seed_for_rand_num_gen, DEBUG=False
    ):
        """InjectTF injected functions class.

        This class contains the implementation of the fault injected functions and
        handles the perturbation of the computed result during runtime in case
        of a fault.

        Args:
            inj_type_scalar (str): The fault injection type for scalar values.
                Refer to `FaultTypes.py` for all implemented fault types.
            inj_type_tensor (str): The fault injection type for tensor values.
                Refer to `FaultTypes.py` for all implemented fault types.
            seed_for_rand_num_gen (int): Seed for the random number generator.
            DEBUG (bool):   Prints debugging information if true. This will print a lot
                to stdout! It is recommended to pipe the output to a log-file.
                Defaults to false.
        """

        InjectedFunctions.__DEBUG = DEBUG

        # get injection function depending on the selected fault type
        InjectedFunctions.__inj_scalar = ftypes.fault_types[inj_type_scalar][0]
        InjectedFunctions.__inj_tensor = ftypes.fault_types[inj_type_tensor][1]

        if seed_for_rand_num_gen:
            np.random.seed(seed_for_rand_num_gen)

    def injection_implemented_for_op(self, op):
        """Returns true if the specified operation `op` has a correspoding
            injected function implemented."""
        return op in InjectedFunctions.__injected_functions

    def get_injected_function(self, op, prob):
        """Returns a fault injected function.

        Args:
            op (tf.Operation): The operation that should be injected.
            prob (float): The probability for injection (value between [0, 1]).
        """
        return InjectedFunctions.__injected_functions[op.type].__func__(prob, op)

    @staticmethod
    def cond_perturb(probability, res):
        """Perturbs the compputed value `res` based on the specified probability.

        Args:
            probability (float): The probability for injection (value between [0, 1]).
            res (float): The value that should be perturbed.

        Returns:
            The perturbed value.
        """

        # random.random returns a number in [0, 1]
        rn = np.random.random()
        if rn <= probability:

            InjectedFunctions.__inject_count += 1

            res = res.numpy()

            # Choose the right injection function depending on if the value
            # is a scalar or a tensor.
            if np.isscalar(res) or (np.ndim(res) == 1 and len(res) == 1):

                res = InjectedFunctions.__inj_scalar(res.dtype, res)
            else:
                res = InjectedFunctions.__inj_tensor(res.dtype, res)

        return res

    @staticmethod
    def get_injection_count():
        """Returns the number of injections that have been performed."""
        return InjectedFunctions.__inject_count

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Implementation of the injected functions.
    #
    # Each static method is essentially a factory for injected functions.
    # The actual injected function is the lambda function that is returned,
    # which is essentially just a wrapper function around the standard
    # TensorFlow function.
    # The static methods accept the probability for injection `prob`
    # as well as the tf.Operation `op` that is to be injected, as arguments.
    # This allows for an easier configuration of the injected
    # function, as its actual arguments are defined by the input(s)
    # to the lambda function.
    # By passing the tf.Operation `op` to the static functions, it is possible to
    # access detailed information that describes the operation, which is neccessary
    # in order to rebuild more complex operations such as Conv2D.
    #
    # Template method:
    # This template demonstrates how the static method should look like
    #
    #  @staticmethod
    #  def __injectedTemplate(prob, op):
    #      return lambda valueX, valueY, valueZ: InjectedFunctions.cond_perturb(
    #          prob, tf.my_Function(valueX, valueY, valueZ)
    #      )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def __injectedAdd(prob, op):
        return lambda a, b: InjectedFunctions.cond_perturb(prob, tf.add(a, b))

    @staticmethod
    def __injectedSub(prob, op):
        return lambda a, b: InjectedFunctions.cond_perturb(prob, tf.subtract(a, b))

    @staticmethod
    def __injectedMul(prob, op):
        return lambda a, b: InjectedFunctions.cond_perturb(prob, tf.multiply(a, b))

    @staticmethod
    def __injectedRelu(prob, op):
        return lambda a: InjectedFunctions.cond_perturb(prob, tf.nn.relu(a))

    @staticmethod  # TODO Test, if done add to __injected_functions dictionary
    def __injectedMatMul(prob, op):
        return lambda a, b: InjectedFunctions.cond_perturb(prob, tf.matmul(a, b))

    @staticmethod  # TODO Test, if done add to __injected_functions dictionary
    def __injectedRsqrt(prob, op):
        return lambda a: InjectedFunctions.cond_perturb(prob, tf.rsqrt(a))

    @staticmethod  # TODO Test, if done add to __injected_functions dictionary
    def __injectedConst(prob, op):
        return lambda a: InjectedFunctions.cond_perturb(prob, tf.constant(a))

    # TODO Test, if done add to __injected_functions dictionary
    # This has not been tested/fully implemented yet. The code shown here
    # is intended to assist further development of the framework.
    # Rebuilding the `Conv2D` operation is not as straightforward as the `Add`
    # operation, but the `op` argument has all necessary information describing
    # the particular operation (in this case `Conv2D`).
    # This should be enough information to build an fault injected operation.
    @staticmethod
    def __injectedConv2D(prob, op):
        return lambda a, b: InjectedFunctions.cond_perturb(
            prob,
            tf.nn.conv2d(
                a,
                filter=b,
                strides=op.get_attr("strides"),
                padding=op.get_attr("padding"),
                use_cudnn_on_gpu=op.get_attr("use_cudnn_on_gpu"),
                data_format=op.get_attr("data_format"),
                dilations=op.get_attr("dilations"),
            ),
        )

    # Dictionary containing all implemented injected functions
    __injected_functions = {
        "Add": __injectedAdd,
        "Sub": __injectedSub,
        "Mul": __injectedMul,
        "Relu": __injectedRelu,
    }
