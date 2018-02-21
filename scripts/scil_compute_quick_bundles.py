#!/usr/bin/env python

from __future__ import print_function

import argparse
import os

import tractconverter as tc
from tractconverter.formats.header import Header

from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric as MDF

DESCRIPTION = """
This script helps using QuickBundles.
    Input: tracts (with the anatomy to load them) and a chosen threshold in mm.

    Output: The centroids from the QuickBundles output.

Notes.
- A very big threshold will keep all the tracts in the same cluster and
  will thus output one centroid, representing the average shape of the tracts.
- A very small threshold will output many clusters. The clusters with the fewest
  tract counts will most likely represent outliers. So far the tract count is
  only printed. Yet to be written.

- QuickBundles require the tracts to have the same number of points. If it is
  not the case, it will perform a subsampling first.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tracts_filename', action='store', metavar='TRACTS', type=str,
                   help='tracts filename. Format must be readable ' +
                        'by the tractconverter.')

    p.add_argument('ref_anat_name', action='store', metavar='ANAT', type=str,
                   help='reference anat to load tracts, and to ' +
                        'save the output.')

    p.add_argument('dist_thresh', action='store', metavar='THRESHOLD', type=float,
                   help='Quickbundles threshold. ' +
                        'See script description for more information.')

    p.add_argument('output_name', action='store', metavar='OUTPUT_NAME', type=str,
                   help='output filename. Format must be ' +
                   'readable by the tractconverter.')

    p.add_argument('--nb_points', dest='nb_of_points', action='store',
                   type=int, default='20',
                   help='QuickBundles require the ' +
                        'tracts to have the same number of points.\nIf it is not ' +
                        'the case, it will perform a subsampling to nb_points ' +
                        'points per tract first. [%(default)s]')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    #####################################
    # Checking if the files exist       #
    #####################################
    for myFile in [args.tracts_filename, args.ref_anat_name]:
        if not os.path.isfile(myFile):
            parser.error('"{0}" must be a file!'.format(myFile))

    if os.path.exists(args.output_name):
        print (args.output_name, " already exist and will be overwritten.")

    #####################################
    # Loading tracts                    #
    #####################################
    tracts_format = tc.detect_format(args.tracts_filename)
    tract_file = tracts_format(args.tracts_filename, anatFile=args.ref_anat_name)
    tracts = [i for i in tract_file]
    hdr = tract_file.hdr

    #####################################
    # Checking if needs subsampling     #
    #####################################
    tmp = len(tracts[0])
    if all(len(my_tract) == tmp for my_tract in tracts):
        nb_of_points = tmp
    else:
        nb_of_points = args.nb_of_points

    #####################################
    # Compute QuickBundles             #
    #####################################
    print('Starting the QuickBundles...')
    # This feature tells QuickBundles to resample each streamlines on the fly.
    feature = ResampleFeature(nb_points=nb_of_points)
    # 'qb' is `dipy.segment.clustering.QuickBundles` object.
    qb = QuickBundles(threshold=args.dist_thresh, metric=MDF(feature))
    # 'clusters' is `dipy.segment.clustering.ClusterMap` object.
    clusters = qb.cluster(tracts)
    centroids = clusters.centroids
    print('    --- done. Number of centroids:', len(centroids))
    print('              Number of points per tract:', nb_of_points)
    print('Cluster sizes:', list(map(len, clusters)))

    #####################################
    # Saving                            #
    #####################################
    print('Saving...')
    out_format = tc.detect_format(args.output_name)
    qb_header = hdr
    qb_header[Header.NB_FIBERS] = len(centroids)
    out_centroids = out_format.create(args.output_name, qb_header,
                                      anatFile=args.ref_anat_name)
    out_centroids += [s for s in centroids]
    out_centroids.close()

    print('    --- done.')

if __name__ == "__main__":
    main()
