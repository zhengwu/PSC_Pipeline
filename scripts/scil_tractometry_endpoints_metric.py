#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
from distutils.version import LooseVersion
import logging
import os

import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
import numpy as np

from scilpy.io.image import assert_same_resolution
from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist,
    assert_outputs_dir_exists_and_empty)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.streamlines import load_in_voxel_space
try:
    from scilpy.tractanalysis.robust_streamlines_metrics import\
           compute_robust_tract_counts_map
except ImportError as e:
    e.args += ("Try running setup.py",)
    raise e


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Projects metrics onto the endpoints of streamlines. The '
                    'idea is to visualize the cortical areas affected by '
                    'metrics (assuming streamlines start/end in the cortex).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument('metrics', nargs='+',
                        help='Nifti metric(s) to compute statistics on')
    parser.add_argument('output_folder',
                        help='Folder where to save endpoints metric')
    add_overwrite_arg(parser)
    return parser


def _compute_streamline_mean(streamline, data):
    # Consider an image size that contains only the streamline. The idea here
    # is to use robust tract count to get the voxels that intersect the
    # streamlines. Could be done with just the vertices, but would not be
    # robust.
    mins = np.min(streamline.astype(int), 0)
    maxs = np.max(streamline.astype(int), 0) + 1
    ranges = maxs - mins
    offset_streamline = streamline - mins

    streamline_density = compute_robust_tract_counts_map(
        [offset_streamline], ranges)
    streamline_data = data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    streamline_average = np.average(streamline_data,
                                    weights=streamline_density)
    return streamline_average


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle] + args.metrics)
    assert_outputs_dir_exists_and_empty(parser, args, args.output_folder)

    metrics = [nib.load(metric) for metric in args.metrics]
    assert_same_resolution(*metrics)

    bundle_tractogram_file = nib.streamlines.load(args.bundle)
    if int(bundle_tractogram_file.header['nb_streamlines']) == 0:
        logging.warning('Empty bundle file {}. Skipping'.format(args.bundle))
        return
    bundle_streamlines_vox = load_in_voxel_space(
        bundle_tractogram_file, metrics[0])

    for metric in metrics:
        data = metric.get_data()
        endpoint_metric_map = np.zeros(metric.shape)
        count = np.zeros(metric.shape)
        for streamline in bundle_streamlines_vox:
            streamline_mean = _compute_streamline_mean(streamline, data)

            xyz = streamline[0, :].astype(int)
            endpoint_metric_map[xyz[0], xyz[1], xyz[2]] += streamline_mean
            count[xyz[0], xyz[1], xyz[2]] += 1

            xyz = streamline[-1, :].astype(int)
            endpoint_metric_map[xyz[0], xyz[1], xyz[2]] += streamline_mean
            count[xyz[0], xyz[1], xyz[2]] += 1

        endpoint_metric_map[count != 0] /= count[count != 0]
        metric_fname, ext = split_name_with_nii(
            os.path.basename(metric.get_filename()))
        nib.save(nib.Nifti1Image(endpoint_metric_map, metric.affine,
                                 metric.header),
                 os.path.join(args.output_folder,
                              '{}_endpoints_metric{}'.format(metric_fname,
                                                             ext)))


if __name__ == '__main__':
    main()
