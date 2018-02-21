#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import assert_outputs_exists, assert_inputs_exist
from scilpy.utils.bvec_bval_tools import normalize_bvecs, is_normalized_bvecs


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Normalize FSL gradient directions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bvecs', help='FSL gradient directions file')
    parser.add_argument('normalized_bvecs',
                        help='Normalized FSL gradient directions file')
    parser.add_argument('-f', action='store_true', dest='overwrite',
                        help="If set, overwrite output file")

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.bvecs])
    assert_outputs_exists(parser, args, [args.normalized_bvecs])

    _, bvecs = read_bvals_bvecs(None, args.bvecs)

    if is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors are already normalized')

    normalize_bvecs(bvecs, args.normalized_bvecs)


if __name__ == '__main__':
    main()
