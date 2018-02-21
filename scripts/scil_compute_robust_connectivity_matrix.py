#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import logging
import os

import nibabel as nib
import numpy as np
import pandas as pd

from scilpy.io.streamlines import load_tracts_over_grid_transition
from scilpy.tractanalysis.tools import intersects_two_rois_atlas
from scilpy.tractanalysis.uncompress import uncompress


def compute_connectivity_matrix(streamlines, atlas_file,
                                allow_self_connection,
                                minimize_self_connection,
                                lower_triangular,
                                normalize):
    atlas_img = nib.load(atlas_file)

    atlas_data = atlas_img.get_data().astype(np.int32)

    real_labels = np.unique(atlas_data)
    nb_real_labels = real_labels.shape[0]

    label_mapping = {real_labels[i]: i for i in range(nb_real_labels)}

    con_mat = np.zeros((nb_real_labels, nb_real_labels), dtype=np.uint32)

    indices = uncompress(streamlines, return_mapping=False)

    for strl_indices in indices:
        logging.debug("\nStarting streamline")
        logging.debug(strl_indices)

        in_label, out_label = intersects_two_rois_atlas(atlas_data,
                                                        strl_indices,
                                                        allow_self_connection,
                                                        minimize_self_connection)

        if in_label is not None and out_label is not None:
            in_index = label_mapping[in_label]
            out_index = label_mapping[out_label]
            con_mat[in_index, out_index] += 1

            if in_label != out_label:
                con_mat[out_index, in_index] += 1

    con_mat = con_mat[1:, 1:]

    if lower_triangular:
        con_mat = np.tril(con_mat)

    if normalize:
        con_mat = con_mat / np.sum(con_mat)

    str_labels = ["{}".format(k) for k in sorted(label_mapping.keys())]
    final_mat = pd.DataFrame(data=con_mat,
                             columns=str_labels[1:],
                             index=str_labels[1:])

    return final_mat


def _buildArgsParser():
    p = argparse.ArgumentParser(
        description='Compute a basic connectivity matrix, robust to compressed '
                    'streamlines. Note that if streamlines are crossing '
                    'more than 2 different labels, the matrix might not be '
                    'perfectly representative, depending on the order of '
                    'traversal of the streamlines.')
    p.add_argument('tracts', metavar='TRACTS',
                   help='path of the tracts file')
    p.add_argument('atlas', metavar='ATLAS',
                   help='path of the nifti file containing an atlas. ' +
                        'The atlas should contain integer labels and 0 for '
                        'unlabeled regions')
    p.add_argument('out', metavar='CON_MAT',
                   help='path of the output connectivity matrix. Should be  '
                        '.csv or .json')
    p.add_argument('--tp', metavar='TRACT_PRODUCER',
                   choices=['scilpy', 'trackvis'],
                   help='tract producer')
    p.add_argument('--no_self_connection', action='store_false',
                   help='if set, will not allow self connections')
    p.add_argument('--no_minimize_self_connection', action='store_false',
                   help='if set, will not attempt to minimize self connections '
                        'and will consider a self connection even if a longer '
                        'range connection could be found.')
    p.add_argument('--lower_triangular', action='store_true',
                   help='if set, will only keep the lower triangular part of '
                        'the connection matrix.')
    p.add_argument('--normalize', action='store_true',
                   help='if set, will normalize the connection matrix.')
    p.add_argument('-f', action='store_true', dest='force',
                   help='Force (overwrite output file).')
    p.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                   help='if set, will log as debug')
    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(args.tracts):
        parser.error('"{0}" must be a file!'.format(args.tracts))

    if os.path.isfile(args.out):
        if args.force:
            logging.info('Overwriting "{0}".'.format(args.out))
        else:
            parser.error(
                '"{0}" already exist! Use -f to overwrite it.'
                .format(args.out))

    out_ext = os.path.splitext(args.out)[1]
    if out_ext not in ['.csv', '.json']:
        parser.error('Unsupported output format. Please use .csv or .json.')

    streamlines = load_tracts_over_grid_transition(args.tracts,
                                                   args.atlas,
                                                   start_at_corner=True,
                                                   tract_producer=args.tp)

    con_mat = compute_connectivity_matrix(streamlines, args.atlas,
                                          args.no_self_connection,
                                          args.no_minimize_self_connection,
                                          args.lower_triangular,
                                          args.normalize)

    if out_ext == '.csv':
        con_mat.to_csv(args.out)
    else:
        con_mat.to_json(args.out, orient='split')


if __name__ == "__main__":
    main()
