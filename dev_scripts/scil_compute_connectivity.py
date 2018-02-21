#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import os
import pickle
import six

from dipy.tracking import utils as dpu

import nibabel as nib
import numpy as np
import tractconverter as tc

from scilpy.utils.streamlines import validate_coordinates


# TODO temp remove this when branch for load all is integrated
def compute_affine_for_dipy_functions(anat, streamlines):
    # Determine if we need to send an identity affine or the real
    # affine. This depends of the space in which streamlines are given by
    # the TractConverter. If we are loading a TCK or TRK file, the streamlines
    # will be aligned with a grid starting at the origin of the reference frame
    # in millimetric space. In that case, send a "scale" identity to density_map
    # to avoid any further transform.
    ref_img = nib.load(anat)
    voxel_dim = ref_img.get_header()['pixdim'][1:4]
    affine_for_dipy = ref_img.get_affine()

    tract_file = streamlines
    if isinstance(streamlines, six.string_types):
        tc_format = tc.detect_format(streamlines)
        tract_file = tc_format(streamlines, anatFile=anat)

    if isinstance(tract_file, tc.formats.tck.TCK) \
       or isinstance(tract_file, tc.formats.trk.TRK):
        affine_for_dipy = np.eye(4)
        affine_for_dipy[:3, :3] *= np.asarray(voxel_dim)

    return affine_for_dipy


def compute_labels_map(lut_fname):
    labels = {}
    with open(lut_fname) as f:
        for line in f:
            tokens = ' '.join(line.split()).split()
            if tokens and not tokens[0].startswith('#'):
                labels[tokens[1]] = tokens[0]

    return labels


def find_mapping(label, label_ids):
    return label_ids.get(label, -1.)


def compute_requested_labels(labels_fname, label_ids):
    mapping = {}

    with open(labels_fname) as f:
        for line in f:
            subparts = line.rstrip().split('_', 1)
            hemisphere = subparts[0]
            name = subparts[1]

            # TODO right now, guess we are 2009
            # Normal Destrieux 2009 syntax
            free_name = 'ctx' + '_' + hemisphere.lower() + '_' + name
            id = find_mapping(free_name, label_ids)

            if id != -1:
                mapping[int(id)] = {'free_name': free_name,
                                    'lut_name': hemisphere + '_' + name}
            else:
                # It was not found. Maybe a deep nuclei, part of Killian?
                # Try it.
                if hemisphere == "LH":
                    label_name = "Left-" + name
                elif hemisphere == "RH":
                    label_name = "Right-" + name

                id = label_ids.get(label_name, -1)
                if id == -1:
                    # Special case for now
                    label_name += "-area"
                    id = label_ids.get(label_name, -1)
                    if id == -1:
                        print("Missing label: {0}".format(subparts[1]))
                    else:
                        mapping[int(id)] = {'free_name': label_name,
                                            'lut_name': hemisphere + '_' + name}
                else:
                    mapping[int(id)] = {'free_name': label_name,
                                        'lut_name': hemisphere + '_' + name}

    return mapping


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Computes the connectivity matrix of a tractogram, using ' +
                    'regions defined\nin a cortical parcellation (typically, ' +
                    'a Freesurfer parcellation).\nThe script also outputs ' +
                    'a mapping file that indicates the mapping\nbetween a row ' +
                    'in the matrix, the basic Freesurfer region name, and\n' +
                    'the names defined in the labels file. This mapping ' +
                    'can then be given to\nplot_matrix.py to customize the ' +
                    'regions\' name display.')

    p.add_argument('tracts', action='store', metavar='TRACTS', type=str,
                   help='name of the tracts file, in a format supported by ' +
                        'the tractconverter')
    p.add_argument('aparc', action='store', metavar='APARC', type=str,
                   help='name of the input aparc file. Currently, this ' +
                        'script is based on aparc.a2009+aseg.')
    p.add_argument('out_matrix', action='store', metavar='OUT', type=str,
                   help='name of the file used to write the connectivity ' +
                        'matrix. Must be saved in the npy format.')
    p.add_argument('out_row_map', action='store', metavar='OUT_MAP', type=str,
                   help='name of the file containing the dictionary mappping ' +
                        'the rows with names. Must be saved in .pkl format.')
    p.add_argument('labels', action='store', metavar='LABELS_TXT', type=str,
                   help='text file listing all wanted labels, following the ' +
                        'format shown in project_scripts/gwen/data/FreeRoisBord.txt' +
                        ', where each line is the name of a region present in ' +
                        'the parcellation.')
    p.add_argument('lut', action='store', metavar='LUT_TXT',  type=str,
                   help='Path of the LUT (normally FreeSurferColorLUT.txt, as' +
                        ' available in project_scripts/gwen/data/).')

    p.add_argument('-f', action='store_true', dest='force_overwrite',
                   default=False, help='Force (overwrite output file). ' +
                                       '[%(default)s]')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error("Tracts file: {0} does not exist.".format(args.tracts))

    # TODO check scilpy supports

    if not os.path.isfile(args.aparc):
        parser.error("Label file: {0} does not exist.".format(args.aparc))

    if not os.path.isfile(args.labels):
        parser.error("Requested region file: {0} does not exist.".format(
            args.labels))

    if not os.path.isfile(args.lut):
        parser.error("Freesurfer LUT file: {0} does not exist.".format(
            args.lut))

    if os.path.isfile(args.out_matrix) and not args.force_overwrite:
        parser.error("Output: {0} already exists. To overwrite, use -f.".format(
            args.out_matrix))

    if os.path.isfile(args.out_row_map) and not args.force_overwrite:
        parser.error("Output: {0} already exists. To overwrite, use -f.".format(
            args.out_row_map))

    if os.path.splitext(args.out_matrix)[1] != ".npy":
        parser.error("Connectivity matrix must be saved in a .npy file.")

    if os.path.splitext(args.out_row_map)[1] != ".pkl":
        parser.error("Mapping must be saved in a .pkl file.")

    # Validate that tracts can be processed
    if not validate_coordinates(args.aparc, args.tracts, nifti_compliant=True):
        parser.error("The tracts file contains points that are invalid.\n" +
                     "Use the remove_invalid_coordinates.py script to clean.")

    # Load labels
    labels_img = nib.load(args.aparc)
    full_labels = labels_img.get_data().astype('int')

    # Compute the mapping from label name to label id
    label_id_mapping = compute_labels_map(args.lut)

    # Find which labels were requested by the user.
    requested_labels_mapping = compute_requested_labels(args.labels,
                                                        label_id_mapping)

    # Filter to keep only needed ones
    filtered_labels = np.zeros(full_labels.shape, dtype='int')
    for label_val in requested_labels_mapping:
        filtered_labels[full_labels == label_val] = label_val

    # Reduce the range of labels to avoid a sparse matrix,
    # because the ids of labels can range from 0 to the 12000's.
    reduced_labels, labels_lut = dpu.reduce_labels(filtered_labels)

    # Load tracts
    tract_format = tc.detect_format(args.tracts)
    tract = tract_format(args.tracts, args.aparc)

    streamlines = [t for t in tract]
    f_streamlines = []
    for sl in streamlines:
        # Avoid streamlines having only one point, as they crash the
        # Dipy connectivity matrix function.
        if sl.shape[0] > 1:
            f_streamlines.append(sl)

    # Compute affine
    affine = compute_affine_for_dipy_functions(args.aparc, args.tracts)

    # Compute matrix
    M = dpu.connectivity_matrix(f_streamlines, reduced_labels, affine=affine,
                                symmetric=True, return_mapping=False,
                                mapping_as_streamlines=False)
    # Remove background connectivity
    M = M[1:, 1:]

    # Save needed files
    np.save(args.out_matrix, np.array(M))

    # Compute the mapping between row numbers, labels and ids.
    sorted_lut = sorted(labels_lut)
    row_name_map = {}
    # Skip first for BG
    for id, lab_val in enumerate(sorted_lut[1:]):
        # Find the associated Freesurfer id
        free_name = requested_labels_mapping[lab_val]['free_name']
        lut_name = requested_labels_mapping[lab_val]['lut_name']

        # Find the mean y position of the label to be able to spatially sort.
        positions = np.where(full_labels == lab_val)
        mean_y = np.mean(positions[1])

        row_name_map[id] = {'free_name': free_name, 'lut_name': lut_name,
                            'free_label': lab_val, 'mean_y_pos': mean_y}

    with open(args.out_row_map, 'w') as f:
        pickle.dump(row_name_map, f)


if __name__ == "__main__":
    main()