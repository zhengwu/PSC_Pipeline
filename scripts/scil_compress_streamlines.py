#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compress tracts by removing collinear (or almost) points.

The compression threshold represents the maximum distance (in mm) to the
original position of the point.
"""

import argparse
import logging

import numpy as np
import tractconverter as tc

try:
    from dipy.tracking.streamlinespeed import compress_streamlines
except ImportError as e:
    raise ImportError("Could not import compress_streamlines from Dipy." +
                      " Do you have a recent enough version?")

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exists,
    check_tracts_same_format, check_tracts_support)


def compression_wrapper(tract_filename, out_filename, error_rate):
    tracts_format = tc.detect_format(tract_filename)
    tracts_file = tracts_format(tract_filename)

    out_hdr = tracts_file.hdr
    out_format = tc.detect_format(out_filename)
    out_tracts = out_format.create(out_filename, out_hdr)

    for s in tracts_file:
        # TODO we should chunk this.
        out_tracts += np.array(compress_streamlines(list([s]), error_rate))

    out_tracts.close()


def _build_args_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'tracts', metavar='TRACTS',
        help='path of the tracts file, in a format supported by the '
             'TractConverter.')
    p.add_argument(
        'out', metavar='OUTPUT_FILE',
        help='path of the output tracts file, in a format supported by the '
             'TractConverter.')

    p.add_argument(
        '-e', dest='errorRate', type=float, default=0.1,
        help='Maximum compression distance in mm. [default: %(default)s]')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.tracts])
    assert_outputs_exists(parser, args, [args.out])
    check_tracts_support(parser, args.tracts, False)
    check_tracts_same_format(parser, args.tracts, args.out)

    if args.errorRate < 0.001 or args.errorRate > 1:
        logging.warn(
            'You are using an error rate of {}.\nWe recommend setting it '
            'between 0.001 and 1.\n0.001 will do almost nothing to the tracts '
            'while 1 will higly compress/linearize the tracts'
            .format(args.errorRate))

    compression_wrapper(args.tracts, args.out, args.errorRate)


if __name__ == "__main__":
    main()
