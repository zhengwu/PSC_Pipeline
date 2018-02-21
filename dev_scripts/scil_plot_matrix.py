#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from mne.viz import circular_layout, plot_connectivity_circle
import numpy as np


def find_mean_y_pos(label_map, label_name):
    for mapping_id in label_map:
        if label_map[mapping_id]['free_name'] == label_name:
            return label_map[mapping_id]['mean_y_pos']


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Displays a connectivity matrix as a connectivity circle.\n\n' +
                    'The names associated to each region can be customized\n' +
                    'using the row_map argument. This row_map is computed\n' +
                    'by the compute_connectivity.py script.')

    p.add_argument('matrix', action='store', metavar='MATRIX', type=str,
                   help='name of the file used containing the connectivity ' +
                        'matrix. Must be in the npy format.')
    p.add_argument('row_map', action='store', metavar='MAP', type=str,
                   help='name of the file containing the dictionary mapping ' +
                        'the rows with names. Must be a .pkl file.')
    p.add_argument('--sort_y', action='store_true',
                   help='if set, labels will be sorted according to their ' +
                        'frontal to occipital order.')
    p.add_argument('--save', action='store', type=str,
                   help='set it to the path to use to save the figure. If ' +
                        'not set, will not save.')
    p.add_argument('--clean_lbl', action='store_true',
                   help='if set, will use the labels given by Bixente. If ' +
                        'not set, will use the default Freesurfer labels.')
    p.add_argument('-f', action='store_true', dest="force",
                   help='force overwriting files, if they exist.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.exists(args.matrix):
        parser.error("{0} is not a valid file path.".format(args.matrix))

    if not os.path.splitext(args.matrix)[1] == ".npy":
        parser.error("Connectivity matrix must be given in .npy format.")

    if not os.path.exists(args.row_map):
        parser.error("{0} is not a valid file path.".format(args.row_map))

    if os.path.splitext(args.row_map)[1] != ".pkl":
        parser.error("Name mapping must be given in a .pkl file.")

    if args.save and os.path.exists(args.save) and not args.force:
        parser.error("Output image: {0} already exists.\n".format(args.save) +
                     "Use -f to force overwriting.")

    con_mat = np.load(args.matrix)

    with open(args.row_map) as f:
        row_name_map = pickle.load(f)

    lh_tags = ['_lh_', 'Left']
    rh_tags = ['_rh_', 'Right']

    if args.clean_lbl:
        label_names = [row_name_map[k]['lut_name'] for k in sorted(row_name_map)]
        lh_tags = ['LH']
        rh_tags = ['RH']
    else:
        label_names = [row_name_map[k]['free_name'] for k in sorted(row_name_map)]
        lh_tags.append('-lh-')
        rh_tags.append('-rh-')

    lh_labels = [name for name in label_names if any(tag in name for tag in lh_tags)]
    rh_labels = [name for name in label_names if any(tag in name for tag in rh_tags)]

    # Validate if all labels were found
    uncut_labels = set(label_names) - set(lh_labels) - set(rh_labels)
    if len(uncut_labels) > 0:
        raise ValueError("Some labels were not filtered as Left or Right.")

    if args.sort_y:
        # TODO choose color
        # label_colors = [label.color for label in labels]
        label_ypos = []
        for name in lh_labels:
            label_ypos.append(find_mean_y_pos(row_name_map, name))
        lh_labels = [label for (ypos, label) in sorted(zip(label_ypos, lh_labels))]
        rh_labels = [label for (ypos, label) in sorted(zip(label_ypos, rh_labels))]

    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)
    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])

    fig, axes = plot_connectivity_circle(con_mat, label_names, linewidth=1.5,
                                   interactive=True, vmin=0, vmax=1,
                                   node_angles=node_angles,
                                   title='All-to-all Connectivity')
    # plot_connectivity_circle(
    #                          node_colors=label_colors,
    #                          fontsize_colorbar=6,

    plt.show(block=True)
    if args.save:
        fig.savefig(args.save, facecolor='black')


if __name__ == "__main__":
    main()
