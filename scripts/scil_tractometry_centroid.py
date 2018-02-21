#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from distutils.version import LooseVersion

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
import nibabel as nib
if LooseVersion(nib.__version__) < LooseVersion('2.1.0'):
    raise ImportError("Unable to import the Nibabel streamline API. "
                      "Nibabel >= v2.1.0 is required")
from nibabel.streamlines.tractogram import Tractogram
import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Compute bundle centroid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bundle', help='Fiber bundle file')
    parser.add_argument('centroid_streamline',
                        help='Output centroid streamline file')
    parser.add_argument(
        '--distance_thres', type=float, default=200,
        help='The maximum distance from a bundle for a streamline to be '
             'still considered as part of it')
    parser.add_argument('--nb_points', type=int, default=20,
                        help='Number of points defining the centroid '
                             'streamline')
    add_overwrite_arg(parser)
    return parser


def get_centroid_streamline(tractogram, nb_points, distance_threshold):
    streamlines = tractogram.streamlines
    resample_feature = ResampleFeature(nb_points=nb_points)
    quick_bundle = QuickBundles(
        threshold=distance_threshold,
        metric=AveragePointwiseEuclideanMetric(resample_feature))
    clusters = quick_bundle.cluster(streamlines)
    centroid_streamlines = clusters.centroids

    if len(centroid_streamlines) > 1:
        raise Exception('Multiple centroids found')

    return Tractogram(centroid_streamlines, affine_to_rasmm=np.eye(4))


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bundle])
    assert_outputs_exists(parser, args, [args.centroid_streamline])
    if args.distance_thres < 0.0:
        parser.error('--distance_thres {} should be '
                     'positive'.format(args.distance_thres))
    if args.nb_points < 2 or args.nb_points > 99:
        parser.error('--nb_points {} should be [2, 99]'
                     .format(args.nb_points))

    tractogram = nib.streamlines.load(args.bundle)
    centroid_streamline = get_centroid_streamline(tractogram,
                                                  args.nb_points,
                                                  args.distance_thres)
    nib.streamlines.save(centroid_streamline,
                         args.centroid_streamline, header=tractogram.header)


if __name__ == '__main__':
    main()
