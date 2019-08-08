#
#   TU-Dresden, Institute of Automation (IfA)
#   Student research thesis
#
#   Evaluation of the effects of common Hardware faults
#   on the accuracy of safety-critical AI components
#
#   Author: Michael Beyer (GitHub: M-Beyer)
#

import unittest

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

tf.enable_eager_execution()

from InjectTF import InjectedFunctions as ifunc


class TestBitFlipInjectedFunctions(unittest.TestCase):
    def setUp(self):
        self.if_bitFlip = ifunc.InjectedFunctions("BitFlip", "BitFlip", None)

    def test_injectedAdd(self):

        # get static methods
        func_BitFlip = self.if_bitFlip._InjectedFunctions__injectedAdd(1.0, None)
        func_NoFlip = self.if_bitFlip._InjectedFunctions__injectedAdd(0.0, None)

        self.assertEqual(tf.add(1.0, 2.0).numpy(), func_NoFlip(1.0, 2.0).numpy())
        self.assertNotEqual(tf.add(1.0, 2.0).numpy(), func_BitFlip(1.0, 2.0).numpy())

    def test_injectedSub(self):

        # get static methods
        func_BitFlip = self.if_bitFlip._InjectedFunctions__injectedSub(1.0, None)
        func_NoFlip = self.if_bitFlip._InjectedFunctions__injectedSub(0.0, None)

        self.assertEqual(tf.subtract(1.0, 2.0).numpy(), func_NoFlip(1.0, 2.0).numpy())
        self.assertNotEqual(
            tf.subtract(1.0, 2.0).numpy(), func_BitFlip(1.0, 2.0).numpy()
        )

    def test_injectedMul(self):

        # get static methods
        func_BitFlip = self.if_bitFlip._InjectedFunctions__injectedMul(1.0, None)
        func_NoFlip = self.if_bitFlip._InjectedFunctions__injectedMul(0.0, None)

        self.assertEqual(tf.multiply(1.0, 2.0).numpy(), func_NoFlip(1.0, 2.0).numpy())
        self.assertNotEqual(
            tf.multiply(1.0, 2.0).numpy(), func_BitFlip(1.0, 2.0).numpy()
        )

    def test_injectedRelu(self):

        # get static methods
        func_BitFlip = self.if_bitFlip._InjectedFunctions__injectedRelu(1.0, None)
        func_NoFlip = self.if_bitFlip._InjectedFunctions__injectedRelu(0.0, None)

        # Testing positive numbers
        self.assertEqual(tf.nn.relu(1.0).numpy(), func_NoFlip(1.0).numpy())
        self.assertNotEqual(tf.nn.relu(1.0).numpy(), func_BitFlip(1.0).numpy())

        self.assertEqual(tf.nn.relu(123.0).numpy(), func_NoFlip(123.0).numpy())
        self.assertNotEqual(tf.nn.relu(123.0).numpy(), func_BitFlip(123.0).numpy())

        # Testing negative numbers
        self.assertEqual(tf.nn.relu(-1.0).numpy(), func_NoFlip(-1.0).numpy())
        self.assertNotEqual(tf.nn.relu(-1.0).numpy(), func_BitFlip(-1.0).numpy())

        self.assertEqual(tf.nn.relu(-123.0).numpy(), func_NoFlip(-123.0).numpy())
        self.assertNotEqual(tf.nn.relu(-123.0).numpy(), func_BitFlip(-123.0).numpy())

        # Testing zero
        self.assertEqual(tf.nn.relu(0.0).numpy(), func_NoFlip(0.0).numpy())
        self.assertNotEqual(tf.nn.relu(0.0).numpy(), func_BitFlip(0.0).numpy())

        # Generate diagram for the injected function as well as the non-injected one
        # -> Visual verification of the ReLU function
        sns.set_style("ticks")

        x1 = np.linspace(-50.0, 50.0, num=100, dtype=np.float32)

        plt.plot(x1, func_NoFlip(x1))
        plt.savefig("./Tests/ReLU_test_not_injected.png", bbox_inches="tight", dpi=300)

        x2_res = np.array([])
        for i in x1:
            val = func_BitFlip(i)
            val = val.numpy()
            x2_res = np.append(x2_res, val)

        plt.yscale("log")
        plt.plot(x1, x2_res)
        plt.savefig("./Tests/ReLU_test_injected.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    unittest.main()
