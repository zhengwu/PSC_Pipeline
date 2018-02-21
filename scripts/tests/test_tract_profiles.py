#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import unittest

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import (
    generate_metrics, generate_streamlines,
    generate_streamlines_with_duplicates)
from scripts.scil_tractometry_tractprofiles import main


class TestTractProfiles(BaseTest):

    def test(self):
        self._test(generate_streamlines(self._tmp_dir), {
            'fake_metric_1': {
                'mean': [1.0] * 5,
                'std': [1.0] * 5,
                'tractprofile': [[[0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0]] * 5]
            },
            'fake_metric_2': {
                'mean': [0.5] * 5,
                'std': [0.5] * 5,
                'tractprofile': [[[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]] * 5]
            }
        })

        self._test(generate_streamlines_with_duplicates(self._tmp_dir), {
            'fake_metric_1': {
                'mean': [1.2] * 5,
                'std': [0.979795897113] * 5,
                'tractprofile': [
                    [[2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0]] * 5]
            },
            'fake_metric_2': {
                'mean': [0.4] * 5,
                'std': [0.489897948557] * 5,
                'tractprofile': [
                    [[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]] * 5]
            }
        })

    def _test(self, bundle_path, gt):
        metrics = generate_metrics(self._tmp_dir)
        bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))

        with RedirectStdOut() as output:
            self.call(main, bundle_path, *metrics, num_points=5)

        output_dict = json.loads('\n'.join(output))
        self.compare_dict_almost_equal(output_dict, {bundle_name: gt})


if __name__ == '__main__':
    unittest.main()
