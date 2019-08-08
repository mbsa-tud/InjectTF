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

from InjectTF import InjectTFUtil as itfutil


class TestNumberConversions(unittest.TestCase):
    def test_float_to_bin32(self):
        self.assertEqual(
            itfutil.float_to_bin32(1.0), "00111111100000000000000000000000"
        )

        self.assertEqual(
            itfutil.float_to_bin32(-1.0), "10111111100000000000000000000000"
        )

        self.assertEqual(itfutil.float_to_bin32(0), "00000000000000000000000000000000")

    def test_bin_to_float32(self):
        self.assertEqual(
            itfutil.bin_to_float32("00111111100000000000000000000000"), 1.0
        )

        self.assertEqual(
            itfutil.bin_to_float32("10111111100000000000000000000000"), -1.0
        )

        self.assertEqual(
            itfutil.bin_to_float32("00000000000000000000000000000000"), 0.0
        )


if __name__ == "__main__":
    unittest.main()
