#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-03-08T15:44:41.042Z
@author: Andy Park
"""

import numpy as np
import flexIk
import numpy.testing as nt
import unittest

class TestVelIkSolver(unittest.TestCase):

    def test_findScaleFactor(self):
        low = -1
        upp = 1
        a = 1
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

        low = -1
        upp = 1
        a = -1
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

        low = -1
        upp = 1
        a = 2
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 0.5)
        self.assertEqual(sMin, 0)

        low = -1
        upp = 1
        a = -2
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 0.5)
        self.assertEqual(sMin, 0)

        low = -1
        upp = 1
        a = 0.5
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

        low = -1
        upp = 1
        a = -0.5
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

        low = 0.4
        upp = 1
        a = 1.6
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 0.625)
        self.assertEqual(sMin, 0.25)

        low = -1.6
        upp = -0.4
        a = -2
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 0.8)
        self.assertEqual(sMin, 0.2)

        low = -1
        upp = 1
        a = 0
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

        low = 0
        upp = 0
        a = 0
        sMax, sMin = flexIk.findScaleFactor(low, upp, a)
        self.assertEqual(sMax, 1)
        self.assertEqual(sMin, 0)

if __name__ == "__main__":

    unittest.main()