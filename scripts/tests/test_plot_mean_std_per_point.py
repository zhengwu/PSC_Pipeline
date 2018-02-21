#!/usr/bin/env python
# -*- coding: utf-8 -*-

from filecmp import dircmp
import os
import unittest

import matplotlib
import numpy as np

matplotlib.use('Agg')

from _BaseTest import BaseTest, RedirectStdOut
from _generate_toy_data import generate_metrics, generate_complex_streamlines
from scripts.scil_tractometry_centroid \
    import main as main_centroids
from scripts.scil_tractometry_label_and_distance_maps \
    import main as main_label_and_distance_maps
from scripts.scil_tractometry_meanstdperpoint \
    import main as main_meanstdperpoint
from scripts.scil_tractometry_plot_meanstdperpoint import main as main_plot
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.metrics_tools import plot_metrics_stats


class TestPlotMeanStdPerPoint(BaseTest):

    def test(self):
        str1, str2 = generate_complex_streamlines(self._tmp_dir)
        self._test(str1, {
            'fake_metric_1': {
                'means': [0.6338513820454637, 0.29622895356515044,
                          0.19617583555085225, 0.5041247563460097,
                          0.60563376069342],
                'stds': [0.9305563870985911, 0.7104268535183597,
                         0.5948669705468858, 0.8683937717021557,
                         0.9189533553424708]
            },
            'fake_metric_2': {
                'means': [0.2860947842290996, 0.31022553821896115,
                          0.18438893015554508, 0.4439592004046368,
                          0.5284218785603157],
                'stds': [0.45193424152857076, 0.4625858338251584,
                         0.3878010476928068, 0.4968495031503124,
                         0.4991915432167323]
            }
        })
        self._test(str2, {
            'fake_metric_1': {
                'means': [0.2764641064762628, 0.1339661758907168,
                          0.17124319099208973, 0.6366605486322581,
                          0.5888568347480626],
                'stds': [0.6902867598200093, 0.49998541528593726,
                         0.5596089273081928, 0.9316568268841207,
                         0.9115707858782105]
            },
            'fake_metric_2': {
                'means': [0.11872122563702849, 0.18038180318721989,
                          0.20944058362836238, 0.681669725683871,
                          0.5820292518901636],
                'stds': [0.323460192636235, 0.3845051472557252,
                         0.4069093579137414, 0.46582841344206033,
                         0.49322530534669456]
            }
        })

    def _test(self, bundle_path, gt):
        centroids_path = os.path.join(self._tmp_dir, 'centroids.trk')
        label_map = os.path.join(self._tmp_dir, 'label.npz')
        distance_map = os.path.join(self._tmp_dir, 'distance.npz')
        metrics = generate_metrics(self._tmp_dir)
        meanstdperpoint_path = os.path.join(
            self._tmp_dir, 'meanstdperpoint.json')

        # We need to create the centroids, label and distance maps, then
        # the mean/std per point in order to test the plot.
        self.call(main_centroids,
                  '-f', bundle_path, centroids_path, nb_points=5)
        self.call(main_label_and_distance_maps,
                  '-f', bundle_path,
                  centroids_path, label_map, distance_map)
        with RedirectStdOut() as output:
            self.call(main_meanstdperpoint,
                      bundle_path, label_map, distance_map, *metrics)

        with open(meanstdperpoint_path, 'w') as meanstdperpoint_file:
            meanstdperpoint_file.writelines(output)

        bundle_name, _ = os.path.splitext(os.path.basename(bundle_path))
        save_plots_to = os.path.join(self._tmp_dir, bundle_name)
        os.mkdir(save_plots_to)
        self.call(main_plot, meanstdperpoint_path, save_plots_to)

        save_plots_gt_to = os.path.join(self._tmp_dir, bundle_name + '_gt')
        os.mkdir(save_plots_gt_to)

        for metric_path in metrics:
            metric, _ = split_name_with_nii(os.path.basename(metric_path))
            fig = plot_metrics_stats(
                np.array(gt[metric]['means']), np.array(gt[metric]['stds']),
                title=bundle_name,
                xlabel='Location along the streamline',
                ylabel=metric)
            fig.savefig(
                os.path.join(save_plots_gt_to,
                             '{}_{}.png'.format(bundle_name, metric)),
                bbox_inches='tight')

        dcmp = dircmp(save_plots_to, save_plots_gt_to)
        if dcmp.diff_files:
            self.failureException()


if __name__ == '__main__':
    unittest.main()
