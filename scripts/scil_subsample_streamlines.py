#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import time

import numpy as np
import tractconverter
from scilpy.tracking.tools import subsample_streamlines


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Subsample a set of streamlines.')
    p.add_argument(
        'input', action='store',  metavar='input',
        type=str,  help='Streamlines input file name.')
    p.add_argument(
        'output', action='store',  metavar='output',
        type=str,  help='Streamlines output file name.')

    p.add_argument(
        '-n', dest='n', action='store', metavar=' ', default=0,
        type=int, help='Maximum number of streamlines to output. [all]')
    p.add_argument(
        '--minL', default=0., type=float,
        help='Minimum length of streamlines. [%(default)s]')
    p.add_argument(
        '--maxL', default=0., type=float,
        help='Maximum length of streamlines. [%(default)s]')
    p.add_argument(
        '--npts', dest='npts', action='store', metavar=' ', default=0, type=int,
        help='Number of points per streamline in the output. [%(default)s]')
    p.add_argument(
        '--arclength', dest='arclength', action='store_true', default=False,
        help='Whether to downsample using arc length parametrization. ' +
             '[%(default)s]')
    p.add_argument('-f', action='store_true', dest='isForce',
                   help='Force (overwrite output file). [%(default)s]')
    p.add_argument('-v', action='store_true', dest='isVerbose',
                   help='Produce verbose output. [%(default)s]')
    p.add_argument(
        '--non_fixed_seed', dest='non_fixed_seed', action='store_true',
        default=False, help='Whether to use a random seed for the subsampling.' +
        ' [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    np.random.seed(int(time.time()))
    in_filename = args.input
    out_filename = args.output
    isForcing = args.isForce
    isVerbose = args.isVerbose

    if isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    if not tractconverter.is_supported(args.input):
        parser.error(
            'Input file must be one of {0}!'
            .format(",".join(tractconverter.FORMATS.keys())))

    inFormat = tractconverter.detect_format(in_filename)
    outFormat = tractconverter.detect_format(out_filename)

    if not inFormat == outFormat:
        parser.error(
            'Input and output must be of the same types!'
            .format(",".join(tractconverter.FORMATS.keys())))

    if os.path.isfile(args.output):
        if args.isForce:
            logging.info('Overwriting "{0}".'.format(out_filename))
        else:
            parser.error(
                '"{0}" already exist! Use -f to overwrite it.'
                .format(out_filename))

    tract = inFormat(in_filename)
    streamlines = [i for i in tract]

    if args.non_fixed_seed:
        rng = np.random.RandomState()
    else:
        rng = None

    results = subsample_streamlines(streamlines, args.minL, args.maxL,
                                    args.n, args.npts, args.arclength, rng)

    logging.info('"{0}" contains {1} streamlines.'.format(
        out_filename, len(results)))

    hdr = tract.hdr
    hdr[tractconverter.formats.header.Header.NB_FIBERS] = len(results)

    output = outFormat.create(out_filename, hdr)
    output += results


if __name__ == "__main__":
    main()
