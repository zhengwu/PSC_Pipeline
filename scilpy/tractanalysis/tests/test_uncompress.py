#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from nibabel.streamlines import Tractogram
import numpy as np

from scilpy.tractanalysis.uncompress import uncompress


class UncompressTestCase(unittest.TestCase):
    def run_test(self):
        tracto = Tractogram(streamlines=[self.strl_data, self.strl_data])
        indices_new = uncompress(tracto.streamlines)
        self.assertTrue(np.allclose(indices_new[0], self.gt_indices))


class BasicUncompressTestCase(UncompressTestCase):
    def setUp(self):
        self.strl_data = np.array([[0.2, 0.2, 0], [1.2, 0.2, 0], [1.5, 2.5, 0],
                                   [2.6, 3.1, 0], [2.9, 3.8, 0], [3.3, 2.8, 0]],
                                  dtype=np.float32)
        self.gt_indices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0],
                                    [2, 2, 0], [2, 3, 0], [3, 3, 0], [3, 2, 0]],
                                   dtype=np.uint16)

    def test(self):
        super(BasicUncompressTestCase, self).run_test()


class LongLineUncompressTestCase(UncompressTestCase):
    def setUp(self):
        self.strl_data = np.array([[0.2, 0.2, 0], [20.2, 0.2, 0]],
                                  dtype=np.float32)
        self.gt_indices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0],
             [6, 0, 0], [7, 0, 0], [8, 0, 0], [9, 0, 0], [10, 0, 0], [11, 0, 0],
             [12, 0, 0], [13, 0, 0], [14, 0, 0], [15, 0, 0], [16, 0, 0],
             [17, 0, 0], [18, 0, 0], [19, 0, 0], [20, 0, 0]], dtype=np.uint16)

    def test(self):
        super(LongLineUncompressTestCase, self).run_test()


class EdgeUncompressTestCase(UncompressTestCase):
    def setUp(self):
        self.strl_data = np.array([[0.2, 0.2, 0], [0.2, 1.2, 0], [0.2, 2.0, 0],
                                   [0.2, 3.1, 0]],
                                  dtype=np.float32)
        self.gt_indices = np.array([[0, 0, 0], [0, 1, 0], [0, 3, 0]],
                                   dtype=np.uint16)

    def test(self):
        super(EdgeUncompressTestCase, self).run_test()


if __name__ == '__main__':
    unittest.main()
