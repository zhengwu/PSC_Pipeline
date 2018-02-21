#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from itertools import chain
import json
import logging
import os

from nibabel.streamlines import load, save, Tractogram
from nibabel.streamlines.trk import TrkFile
import numpy as np

from scilpy.utils.streamlines import perform_streamlines_operation
from scilpy.utils.streamlines import subtraction, intersection, union

OPERATIONS = {
    'subtraction': subtraction,
    'intersection': intersection,
    'union': union,
}

DESCRIPTION = """
Performs an operation on a list of streamline files. The supported
operations are:

    subtraction:  Keep the streamlines from the first file that are not in
                  any of the following files.

    intersection: Keep the streamlines that are present in all files.

    union:        Keep all streamlines while removing duplicates.

For efficiency, the comparisons are performed using a hash table. This means
that streamlines must be identical for a match to be found. To allow a soft
match, use the --precision option to round streamlines before processing.
Note that the streamlines that are saved in the output are the original
streamlines, not the rounded ones.

The metadata (data per point, data per streamline) of the streamlines that
are kept in the output will preserved. This requires that all input files
share the same type of metadata. If this is not the case, use the option
--no-data to strip the metadata from the output.

"""


def build_args_parser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    parser.add_argument('operation', action='store',
                        choices=OPERATIONS.keys(),
                        metavar='OPERATION', type=str,
                        help='The type of operation to be performed on the '
                             'streamlines. Must\nbe one of the following: '
                             '%(choices)s.')

    parser.add_argument('inputs', action='store',
                        metavar='INPUT_FILES', type=str, nargs='+',
                        help='The list of files that contain the ' +
                             'streamlines to operate on.')

    parser.add_argument('output', action='store',
                        metavar='OUTPUT_FILE', type=str,
                        help='The file where the remaining streamlines '
                             'are saved.')

    parser.add_argument('--precision', '-p', action='store',
                        metavar='NUMBER_OF_DECIMALS', type=int,
                        help='The precision used when comparing streamlines.')

    parser.add_argument('--no-data', '-n', action='store_true',
                        help='Strip the streamline metadata from the output.')

    parser.add_argument('--save-meta-indices', '-m', action='store_true',
                        help='Save streamline indices to metadata. Has no '
                             'effect if --no-data\nis present. Will '
                             'overwrite \'ids\' metadata if already present.')

    parser.add_argument('--save-indices', '-s', action='store',
                        metavar='OUTPUT_INDEX_FILE', type=str,
                        help='Save the streamline indices to the supplied '
                             'json file.')

    parser.add_argument('--verbose', '-v', action='store_true', dest='verbose',
                        help='Produce verbose output.')

    parser.add_argument('--force', '-f', action='store_true', dest='force',
                        help='Overwrite output file if present.')

    return parser


def load_data(path):
    logging.info(
        'Loading streamlines from {0}.'.format(path))
    tractogram = load(path).tractogram
    streamlines = tractogram.streamlines
    data_per_streamline = tractogram.data_per_streamline
    data_per_point = tractogram.data_per_point

    return streamlines, data_per_streamline, data_per_point


def main():

    parser = build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if os.path.isfile(args.output):
        if args.force:
            logging.info('Overwriting {0}.'.format(args.output))
        else:
            parser.error(
                '{0} already exist! Use -f to overwrite it.'
                .format(args.output))

    # Load all input streamlines.
    data = [load_data(f) for f in args.inputs]
    streamlines, data_per_streamline, data_per_point = zip(*data)
    nb_streamlines = [len(s) for s in streamlines]

    # Apply the requested operation to each input file.
    logging.info(
        'Performing operation \'{}\'.'.format(args.operation))
    new_streamlines, indices = perform_streamlines_operation(
        OPERATIONS[args.operation], streamlines, args.precision)

    # Get the meta data of the streamlines.
    new_data_per_streamline = {}
    new_data_per_point = {}
    if not args.no_data:

        for key in data_per_streamline[0].keys():
            all_data = np.vstack([s[key] for s in data_per_streamline])
            new_data_per_streamline[key] = all_data[indices, :]

        # Add the indices to the metadata if requested.
        if args.save_meta_indices:
            new_data_per_streamline['ids'] = indices

        for key in data_per_point[0].keys():
            all_data = list(chain(*[s[key] for s in data_per_point]))
            new_data_per_point[key] = [all_data[i] for i in indices]

    # Save the indices to a file if requested.
    if args.save_indices is not None:
        start = 0
        indices_dict = {'filenames': args.inputs}
        for name, nb in zip(args.inputs, nb_streamlines):
            end = start + nb
            file_indices = \
                [i - start for i in indices if i >= start and i < end]
            indices_dict[name] = file_indices
            start = end
        with open(args.save_indices, 'wt') as f:
            json.dump(indices_dict, f)

    # Save the new streamlines.
    logging.info('Saving streamlines to {0}.'.format(args.output))
    reference_file = load(args.inputs[0], True)
    new_tractogram = Tractogram(
        new_streamlines, data_per_streamline=new_data_per_streamline,
        data_per_point=new_data_per_point)

    # If the reference is a .tck, the affine will be None.
    affine = reference_file.tractogram.affine_to_rasmm
    if affine is None:
        affine = np.eye(4)
    new_tractogram.affine_to_rasmm = affine

    new_header = reference_file.header.copy()
    new_header['nb_streamlines'] = len(new_streamlines)
    save(new_tractogram, args.output, header=new_header)


if __name__ == "__main__":
    main()
