#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os

import tractconverter as tc

from scilpy.utils.streamlines import get_tract_count


def _buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Count the number of tracts in a streamlines file.')
    p.add_argument('tracts', action='store',
                   metavar='TRACTS', type=str,
                   help='path of the tracts file, in a format supported by ' +
                        'the TractConverter.')
    p.add_argument('--out', action='store',
                   metavar='OUTPUT_FILE', type=str,
                   help='path of the output text file. ' +
                        'If not given, will print to stdout')
    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if not tc.is_supported(args.tracts):
        parser.error('Format of "{0}" not supported.'.format(args.tracts))

    tract_count = get_tract_count(args.tracts)

    if args.out:
        with open(args.out) as f:
            f.write(str(tract_count))
    else:
        print(tract_count)


if __name__ == "__main__":
    main()
