#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_centroid import main


class TestCentroids(BaseTest):

    def test(self):
        save_centroids_to = os.path.join(self._tmp_dir, 'centroids.trk')
        centroids_gt = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
        for generate in (generate_streamlines,
                         generate_streamlines_with_duplicates):
            self.call(main,
                      generate(self._tmp_dir),
                      save_centroids_to,
                      nb_points=5)
            self.compare_streamlines(
                save_centroids_to, centroids_gt)
            os.remove(save_centroids_to)


if __name__ == '__main__':
    unittest.main()
