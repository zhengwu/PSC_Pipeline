#!/usr/bin/env python

from __future__ import print_function

import argparse
import os

import numpy as np
import tractconverter as tc
import nibabel as nib

from dipy.tracking.distances import cut_plane


def compute_cross_sections(streamlines, ref_streamline):
    print('Computing sections...')
    if not isinstance(streamlines, list):  # To make sure it is as a list.
        streamlines = [streamlines]
    hits = cut_plane(streamlines, ref_streamline)
    hitpoints = [h[:, :3] for h in hits]
    # angles = [h[:, 3:] for h in hits] #We don't use it.
    print('---done')
    return hitpoints


def compute_stats(hitpoints, anat, affine, out_name, planes_for_stats):
    print('Creating map and computing stats...')

    #####################################
    # Preparing variables               #
    #####################################
    map3d = np.zeros((anat.shape[0], anat.shape[1], anat.shape[2]))
    nb_planes = len(hitpoints)
    nb_voxels = np.zeros(nb_planes)

    #####################################
    # Computing statistics and map      #
    #####################################
    for this_plane in range(nb_planes):
        these_hitpoints = np.asarray(hitpoints[this_plane])
        nb_these_hitpoints = len(these_hitpoints)
        for p in range(nb_these_hitpoints):
            this_point = these_hitpoints[p]
            this_indice = np.round(this_point).astype(int)  # The interpolation should
            # already be done by Dipy
            if anat[this_indice[0], this_indice[1], this_indice[2]] != 0:
                map3d[this_indice[0], this_indice[1], this_indice[2]] = this_plane + 1
                # Labels: 1 to nb_sections
        nb_voxels[this_plane] = len(np.where(map3d == this_plane + 1)[0])
        # Note: if two planes cut each other, still ok, because the counted plane was the last one.

    if out_name:
        print(' ... saving labels map')
        nib.save(nib.Nifti1Image(map3d.astype('float32'), affine), out_name)

    #####################################
    # Printing stats                    #
    #####################################
    print(' ... nb of planes: ', nb_planes)
    if planes_for_stats:
        first_plane = max(1, planes_for_stats[0])
        last_plane = min(planes_for_stats[1], nb_planes)
        print('Printing stats from plane', first_plane, 'to plane', last_plane, 'out of', nb_planes)
    else:
        first_plane = 1
        last_plane = nb_planes

    nb_voxels_includedPlanes = nb_voxels[first_plane - 1: last_plane]
    smallest_plane = nb_voxels_includedPlanes.argmin() + first_plane
    biggest_plane = nb_voxels_includedPlanes.argmax() + first_plane
    print('      Smallest (', smallest_plane, '):', nb_voxels[smallest_plane-1])
    print('      Biggest (', biggest_plane, '):', nb_voxels[biggest_plane-1])
    print('      Average:', np.mean(nb_voxels_includedPlanes))


DESCRIPTION = """
This script follows the direction given by a reference streamline, and, at each
point on this streamline (except the first and last), counts statistics on the
section perpendicular to the direction of the streamline at this point,
considering a tracts file, such as the number of voxels crossed by fibers.

The reference streamline could be obtained by using compute_QuickBundles with a
very large threshold, for instance. It is thus called Centroid, here.

      Input: the tracts file, the reference streamline (centroid).

      Output: no output. Prints the statistics.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tracts_filename', action='store', metavar='TRACTS', type=str,
                   help='tracts filename. Format must be readable ' +
                        'by the tractconverter.')

    p.add_argument('centroid_filename', action='store', metavar='CENTROID', type=str,
                   help='The reference streamline could be ' +
                        'obtained by using compute_QuickBundles\nwith a very large ' +
                        'threshold, for instance.')

    p.add_argument('ref_anat_name', action='store', metavar='ANAT', type=str,
                   help='Anatomy to load fibers.')

    p.add_argument('--output_name', dest='output_name', action='store',
                   metavar=' ', type=str,
                   help='Output name for the maps with the number of the sections ' +
                        '(from 1 to nb_planes)\nas label for the voxels (format: nifti)' +
                        '. If two sections cut each other,\nthe highest label is used ' +
                        '(but the voxel is counted twice in the statistics).')

    p.add_argument('--planes_for_stats', dest='planes_for_stats',
                   action='store', metavar=' ', type=str,
                   help='Planes to include in stats. format: (first,last).' +
                        ' Planes number start at 1.')

    p.add_argument('--mask', action='store_true',
                   help='The given anat will be used as a mask, i.e. only ' +
                        'voxels in the mask will be counted in statistics')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    #####################################
    # Applying options                  #
    #####################################
    if args.planes_for_stats:
        planes_for_stats = eval(args.planes_for_stats)
    else:
        planes_for_stats = None

    #####################################
    # Checking if the files exist       #
    #####################################
    for myFile in [args.tracts_filename, args.centroid_filename, args.ref_anat_name]:
        if not os.path.isfile(myFile):
            parser.error('"{0}" must be a file!'.format(myFile))

    if args.output_name:
        if os.path.exists(args.output_name):
            print (args.output_name, " already exist and will be overwritten.")

    #####################################
    # Loading tracts                    #
    #####################################
    tract_format = tc.detect_format(args.tracts_filename)
    tract = tract_format(args.tracts_filename, anatFile=args.ref_anat_name)
    streamlines = [i for i in tract]

    centroid_format = tc.detect_format(args.centroid_filename)
    tmp = centroid_format(args.centroid_filename,  anatFile=args.ref_anat_name)
    centroid = [i for i in tmp]  # should contain only one
    if len(centroid) > 1:
        print('Centroid should contain only one streamline. Here, the file contains more.')
        print('The first streamline will be used as the centroid.')
    centroid = centroid[0]

    #####################################
    # Loading anat                      #
    # Preparing mask                    #
    #####################################
    anat = nib.load(args.ref_anat_name)
    affine = anat.get_affine()
    shape = anat.get_shape()
    if args.mask:
        anat = anat.get_data()
    else:
        anat = np.ones(shape)

    hitpoints = compute_cross_sections(streamlines, centroid)
    compute_stats(hitpoints, anat, affine, args.output_name, planes_for_stats)


if __name__ == "__main__":
    main()
