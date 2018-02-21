#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt
import os
import unittest

import numpy as np

from _BaseTest import BaseTest
from _generate_toy_data import (
    generate_streamlines, generate_streamlines_with_duplicates)
from scripts.scil_tractometry_centroid import main as main_centroids
from scripts.scil_tractometry_label_and_distance_maps \
    import main as main_label_and_distance_maps


class TestLabelAndDistanceMaps(BaseTest):

    def test(self):
        centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        output_label = os.path.join(self._tmp_dir, 'label.npz')
        output_distance = os.path.join(self._tmp_dir, 'distance.npz')

        for generate, nb_streamlines, distance_gt in (
                (generate_streamlines, 8, ([sqrt(8)] * 20) + ([sqrt(2)] * 20)),
                (generate_streamlines_with_duplicates, 10,
                 ([sqrt(8)] * 30) + ([sqrt(2)] * 20))):
            bundle_path = generate(self._tmp_dir)
            bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

            # We first need to create the centroids
            self.call(main_centroids,
                      bundle_path, centroids_path, nb_points=5)

            self.call(main_label_and_distance_maps,
                      bundle_path,
                      centroids_path,
                      output_label,
                      output_distance)

            label_gt = [1, 2, 3, 4, 5] * nb_streamlines
            self.assertEqual(
                list(np.load(output_label)['arr_0']),
                label_gt,
                "Wrong label map for {}.".format(bundle_name))

            tested_distance = np.load(output_distance)['arr_0']
            if not np.allclose(tested_distance, distance_gt):
                raise self.failureException("Wrong distance map for {}."
                                            .format(bundle_name))

            for f in [centroids_path, output_label, output_distance]:
                os.remove(f)


if __name__ == '__main__':
    unittest.main()
