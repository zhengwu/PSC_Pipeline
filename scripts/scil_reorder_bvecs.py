#! /usr/bin/env python

import argparse

from scilpy.io.utils import (assert_outputs_exists, assert_inputs_exist)
from scilpy.utils.bvec_bval_tools import reorder_bvecs_fsl, reorder_bvecs_mrtrix
from scilpy.utils.util import str_to_index


def build_args_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Reorder the bvecs.')

    parser.add_argument('gradients_file', action='store',
                        metavar='gradients_file', type=str,
                        help='Path to the gradients file.')
    parser.add_argument('axes', action='store', metavar='axes', type=str,
                        help='New ordering of axes. eg: to swap the z and y'
                             ' axes, use: xzy')
    parser.add_argument('reordered_gradients_file', action='store',
                        metavar='reordered_gradients_file', type=str,
                        help='Path to the reordered gradient file.')
    parser.add_argument('-f', action='store_true', dest='overwrite',
                        help='Force (overwrite output file). [%(default)s]')

    # TODO Remove this param when the unified gradient loader is in use.
    gradients_type = parser.add_mutually_exclusive_group(required=True)
    gradients_type.add_argument('--fsl', dest='fsl_bvecs', action='store_true',
                                help='Specify fsl format')
    gradients_type.add_argument('--mrtrix', dest='fsl_bvecs',
                                action='store_false',
                                help='Specify mrtrix format')

    return parser


def main():
    parser = build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.gradients_file])
    assert_outputs_exists(parser, args, [args.reordered_gradients_file])

    indices = [str_to_index(axis) for axis in list(args.axes)]
    if len(indices) != 3 or {0, 1, 2} != set(indices):
        parser.error('The axes parameter must contain x, y and z in whatever '
                     'order.')

    if args.fsl_bvecs:
        reorder_bvecs_fsl(args.gradients_file, indices,
                          args.reordered_gradients_file)
    else:
        reorder_bvecs_mrtrix(args.gradients_file, indices,
                             args.reordered_gradients_file)

if __name__ == "__main__":
    main()
