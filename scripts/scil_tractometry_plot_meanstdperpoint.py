#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import os

import numpy as np

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist,
    assert_outputs_dir_exists_and_empty)
from scilpy.utils.metrics_tools import plot_metrics_stats
from scilpy.utils.python_tools import natural_sort


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Plot mean/std per point',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('meanstdperpoint',
                        help='JSON file containing the mean/std per point')
    parser.add_argument('output', help='Output directory')

    parser.add_argument(
        '--fill_color',
        help='Hexadecimal RGB color filling the region between mean ± std. '
             'The hexadecimal RGB color should be formatted as 0xRRGGBB')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.meanstdperpoint])
    assert_outputs_dir_exists_and_empty(parser, args, args.output)

    if args.fill_color and len(args.fill_color) != 8:
        parser.error('Hexadecimal RGB color should be formatted as 0xRRGGBB')

    with open(args.meanstdperpoint, 'r+') as f:
        meanstdperpoint = json.load(f)

    for bundle_name, bundle_stats in meanstdperpoint.iteritems():
        for metric, metric_stats in bundle_stats.iteritems():
            labels = natural_sort(metric_stats.keys())
            means = [metric_stats[label]['mean'] for label in labels]
            stds = [metric_stats[label]['std'] for label in labels]

            fig = plot_metrics_stats(
                np.array(means), np.array(stds),
                title=bundle_name,
                xlabel='Location along the streamline',
                ylabel=metric,
                fill_color=(args.fill_color.replace("0x", "#")
                            if args.fill_color else None))
            fig.savefig(
                os.path.join(args.output,
                             '{}_{}.png'.format(bundle_name, metric)),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
