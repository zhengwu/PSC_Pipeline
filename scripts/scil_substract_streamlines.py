#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
import logging
import os

import tractconverter

from scilpy.utils.streamlines import substract_streamlines

def build_args_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='WARNING: The script scil_substract_streamlines is '
                    'deprecated. Use \nscil_streamlines_math instead. '
                    '\n\nSubstract (remove) streamlines from a file.')

    parser.add_argument('input', action='store',
                        metavar='INPUT_TRACTS', type=str,
                        help='The file from which streamlines are removed.')

    parser.add_argument('remove', action='store',
                        metavar='TRACTS_TO_REMOVE', type=str, nargs='+',
                        help='The list of files that contain the ' +
                             'streamlines to remove.')

    parser.add_argument('output', action='store',
                        metavar='OUTPUT_TRACTS', type=str,
                        help='The file where the remaining files are saved.')

    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='verbose output')

    parser.add_argument('-f', action='store_true', dest='force',
                        help='force (overwrite output file if present)')

    return parser


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    logging.warning('The script scil_substract_streamlines is deprecated. '
                    'Use scil_streamlines_math instead.')

    if os.path.isfile(args.output):
        if args.force:
            logging.info('Overwriting {0}.'.format(args.output))
        else:
            parser.error(
                '{0} already exist! Use -f to overwrite it.'
                .format(args.output))

    # The first filename contains the streamlines from which all others are
    # substracted.
    logging.info(
        'Loading streamlines from file {0} ...'.format(args.input))
    tract_format = tractconverter.detect_format(args.input)
    streamlines = list(tract_format(args.input))

    # All the other filenames contain the streamlines to be removed.
    streamlines_to_remove = []
    for filename in args.remove:
        logging.info(
            'Loading streamlines from file {0} ...'.format(filename))
        tract_format = tractconverter.detect_format(filename)
        streamlines_to_remove.append(tract_format(filename))

    # Remove the streamlines in place.
    substract_streamlines(
        streamlines,
        itertools.chain(*streamlines_to_remove))

    # Save the new streamlines.
    logging.info('Saving remaining streamlines ...')
    tract_format = tractconverter.detect_format(args.input)
    input_tract = tract_format(args.input)
    hdr = input_tract.hdr
    hdr[tractconverter.formats.header.Header.NB_FIBERS] = len(streamlines)
    output = tract_format.create(args.output, hdr)
    output += streamlines


if __name__ == "__main__":
    main()
