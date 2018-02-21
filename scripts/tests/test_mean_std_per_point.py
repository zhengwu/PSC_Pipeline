#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_metrics, generate_streamlines,
    generate_streamlines_with_duplicates)
from scripts.scil_tractometry_centroid \
    import main as main_centroids
from scripts.scil_tractometry_label_and_distance_maps \
    import main as main_label_and_distance_maps
from scripts.scil_tractometry_meanstdperpoint \
    import main as main_meanstdperpoint


class TestMeanStdPerPoint(BaseTest):

    def setUp(self):
        super(TestMeanStdPerPoint, self).setUp()
        self.centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        self.label_map = os.path.join(self._tmp_dir, 'label.npz')
        self.distance_map = os.path.join(self._tmp_dir, 'distance.npz')
        self.metrics = generate_metrics(self._tmp_dir)
        self.labels = ['01', '02', '03', '04', '05']

    def _create_maps(self, bundle_path):
        # We first need to create the centroids
        self.call(main_centroids,
                  '-f', bundle_path, self.centroids_path, nb_points=5)

        # Then, we need to create the label and distance maps
        self.call(main_label_and_distance_maps,
                  '-f', bundle_path,
                  self.centroids_path, self.label_map, self.distance_map)

    def _compare(self, bundle_path, gt, output):
        bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))
        output_dict = json.loads('\n'.join(output))
        self.compare_dict_almost_equal(output_dict[bundle_name], gt)

    def test_no_weighting(self):
        fake1 = {'mean': 1.0, 'std': 1.0}
        fake2 = {'mean': 0.5, 'std': 0.5}

        bundle_path = generate_streamlines(self._tmp_dir)
        self._create_maps(bundle_path)
        gt = {'fake_metric_1': {l: fake1 for l in self.labels},
              'fake_metric_2': {l: fake2 for l in self.labels}}

        with RedirectStdOut() as output:
            self.call(main_meanstdperpoint,
                      bundle_path, self.label_map, self.distance_map,
                      *self.metrics)

        self._compare(bundle_path, gt, output)

    def test_distance_weighting(self):
        fake1 = {'mean': 2.0 / 3.0, 'std': 0.942809041582}
        fake2 = {'mean': 2.0 / 3.0, 'std': 0.471404520791}
        bundle_path = generate_streamlines(self._tmp_dir)
        self._create_maps(bundle_path)
        gt = {'fake_metric_1': {l: fake1 for l in self.labels},
              'fake_metric_2': {l: fake2 for l in self.labels}}

        with RedirectStdOut() as output:
            self.call(main_meanstdperpoint,
                      '--distance_weighting',
                      bundle_path, self.label_map, self.distance_map,
                      *self.metrics)

        self._compare(bundle_path, gt, output)

    def test_distance_density_weighting(self):
        fake1 = {'mean': 1.11111111111, 'std': 0.99380799}
        fake2 = {'mean': 0.444444444444, 'std': 0.496903995}
        bundle_path = generate_streamlines_with_duplicates(self._tmp_dir)
        self._create_maps(bundle_path)
        gt = {'fake_metric_1': {l: fake1 for l in self.labels},
              'fake_metric_2': {l: fake2 for l in self.labels}}

        with RedirectStdOut() as output:
            self.call(main_meanstdperpoint,
                      '--distance_weighting', '--density_weighting',
                      bundle_path, self.label_map, self.distance_map,
                      *self.metrics)

        self._compare(bundle_path, gt, output)


if __name__ == '__main__':
    unittest.main()
