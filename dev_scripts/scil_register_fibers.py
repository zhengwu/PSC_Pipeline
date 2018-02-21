#! /usr/bin/env python

from __future__ import print_function

import argparse
import os

import nibabel as nib
import numpy as np

import tractconverter as tc

DESCRIPTION = """
Script to apply a linear registration matrix on fibers (that is, on every point
of the fibers). Matrix should be written as a 4-line text file with 4 numbers on
each line, separated by a space.

     Input: The fibers, an exemple of initial and final reference anatomy,
     necessary to load and save fibers, and the registration matrix.

     Output: The registered fibers.

Still to be verified: if the script works when saving tracts as .tck.
Still a nested loop. Could be done with matrices?
"""

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('tracts_name', action='store', metavar='INPUT_TRACTS',
                   type=str, help='tracts filename. May be in any format' +
                   'readable by the tractconverter.')

    p.add_argument('init_anat_name', action='store', metavar='anat_init',
                   type=str, help='Reference anatomy (initial space),' +
                   'necessary to load the fibers')

    p.add_argument('matrix_name', action='store', metavar='MATRIX',
                   type=str, help='Transformation matrix, as a text file,' +
                   ' ex: .txt, .mat., written as FLIRT writes them.')

    p.add_argument('output_name', action='store', metavar='OUTPUT_NAME',
                   type=str, help='Name of the output file, containing the' +
                   'registered fibers. May be in any format readable by' +
                   'the tractometer. TCK OK????')

    p.add_argument('final_anat_name', action='store', metavar='anat_final',
                   type=str, help='Reference anatomy (final space),' +
                   'necessary to save the fibers.')

    return p

def register(init_shape, nb_tracts, tracts_init, Rot, Trans):
    tracts_final = []
    for tract in range(nb_tracts):
        nb_points = len(tracts_init[tract])
        tracts_final.append(np.zeros((nb_points,3)))
        for point in range(nb_points):
            indice = tracts_init[tract][point]
            indice_flip = np.asarray([init_shape[0] - indice[0], indice[1], indice[2]]) #flipping
            temp = np.dot(Rot, [indice_flip[0], indice_flip[1], indice_flip[2]]) + Trans
            temp = np.asarray([init_shape[0]-temp[0], temp[1], temp[2]]) #flipping back
            tracts_final[tract][point] = temp

    return tracts_final

def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    print( 'Fiber registration...')

    #####################################
    # Checking if the files exist       #
    #####################################
    for myFile in [args.init_anat_name, args.final_anat_name, args.matrix_name, args.tracts_name]:
        if not os.path.isfile(myFile):
            parser.error('"{0}" must be a file!'.format(myFile))

    if os.path.exists(args.output_name):
        print (args.output_name, " already exist and will be overwritten.")

    #####################################
    # Loading anatomies                 #
    # Applying voxel sizes              #
    #####################################
    init_anat_img = nib.load(args.init_anat_name)
    init_shape = np.array(init_anat_img.shape) * np.array(init_anat_img.get_header()['pixdim'][1:4])

    #####################################
    # Loading tracts                    #
    #####################################
    tracts_format = tc.detect_format(args.tracts_name)
    tracts_file = tracts_format(args.tracts_name, anatFile=args.init_anat_name )
    hdr = tracts_file.hdr
    tracts_init = [i for i in tracts_file]

    nb_tracts = len(tracts_init)

    #####################################
    # Loading the matrix                #
    #####################################
    Big_matrix = np.loadtxt(args.matrix_name)
    Rot= Big_matrix[0:3,0:3]
    Trans = Big_matrix[0:3,3]

    #####################################
    # Registration                      #
    #####################################
    #tracts_final = register(init_shape, nb_tracts, tracts_init, Rot, Trans)
    tracts_final = []
    for this_tract in range(nb_tracts):
        nb_points_in_tract = len(tracts_init[this_tract])
        tracts_final.append(np.zeros((nb_points_in_tract,3)))
        for this_point in range(nb_points_in_tract):
            indice = tracts_init[this_tract][this_point]
            indice_flip = np.asarray([init_shape[0] - indice[0], indice[1], indice[2]]) #flipping
            indice_registered = np.dot(Rot, [indice_flip[0], indice_flip[1], indice_flip[2]]) + Trans
            indice_registered = np.asarray([init_shape[0]-indice_registered[0], indice_registered[1], indice_registered[2]]) #flipping back
            tracts_final[this_tract][this_point] = indice_registered

    #####################################
    # Saving                             #
    #####################################
    out_format = tc.detect_format(args.output_name)
    out_tracts = out_format.create(args.output_name, hdr, anatFile=args.final_anat_name)
    out_tracts += [t for t in tracts_final] # a tester

    out_tracts.close()
    print('...Done')


if __name__ == "__main__":
    main()