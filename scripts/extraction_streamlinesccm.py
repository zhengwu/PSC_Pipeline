#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import six
import argparse
import pickle
import pdb

import scipy.io as sio
import nibabel as nib
import numpy as np
import tractconverter as tc
from dipy.tracking import utils as dpu
from dipy.tracking.metrics import downsample
from dipy.segment.clustering import QuickBundles

from scilpy.utils.streamlines import validate_coordinates
from scilpy.connectome.fibers_processing_functions import cortexband_dilation_wm,nconnectivity_matrix
from dipy.tracking.streamline import length


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

            #Desikan-Killian
            if id == -1:
                free_name = 'ctx' + '-' + hemisphere.lower() + '-' + name
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
                elif hemisphere == "":
                    label_name = name

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
        description='Extract the streamline connectivity cell matrix from input streamlines.\n\n' +
                    'The input includes the streamlines, the parcellation of the brain, \n' +
                    'labels and lookup table. \n' + 
                    'The output is the streamline connectivity matrix. \n')

    p.add_argument('tracts', action='store', metavar='TRACTS', type=str,
                   help='name of the tracts file, in a format supported by ' +
                        'the tractconverter')

    p.add_argument('aparc', action='store', metavar='APARC', type=str,
                   help='name of the input aparc file. Currently, this ' +
                        'script is based on freesurfer output: wmparc and aparc.a2009+aseg.')


    p.add_argument('labels', action='store', metavar='LABELS_TXT', type=str,
                   help='text file listing all wanted labels, following the ' +
                        'format shown in project_scripts/scilpy/connectome/xxxx_ROI.txt' +
                        ', where each line is the name of a region present in ' +
                        'the parcellation.')

    p.add_argument('lut', action='store', metavar='LUT_TXT',  type=str,
                   help='Path of the LUT (normally FreeSurferColorLUT.txt, as' +
                        ' available in project_scripts/scilpy/connectome/).')

    p.add_argument('sub_id', action='store', metavar='SUB_ID', type=str,
                   help='subject id, used for naming the output.')

    p.add_argument('minlen', action='store', metavar='MINLEN', type=float, default=20,
                   help='minimum length for fibers in the streamline cm.')

    p.add_argument('maxlen', action='store', metavar='MAXLEN', type=float, default=200,
                   help='minimum length for fibers in the streamline cm.')

    p.add_argument('dilation_dist', action='store', metavar='DILATION_DIST', type=int, default=2,
                   help='distance (in voxel) for WM dilation.')

    p.add_argument('dilation_windsize', action='store', metavar='DILATION_WINDSIZE', type=int, default=2,
                   help='window size (in voxel) for WM dilation.')

    p.add_argument('cnpoint', action='store', metavar='CNPOINT', type=float, default=5,
                   help='number of points in each ROI to be considered as connection.')

    p.add_argument('pre', action='store', metavar='PRE',  type=str,
                   help='prefix for the name of saved files.')



    return p

def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.tracts):
        parser.error("Tracts file: {0} does not exist.".format(args.tracts))

    if not os.path.isfile(args.aparc):
        parser.error("Label file: {0} does not exist.".format(args.aparc))
    
    if not os.path.isfile(args.labels):
        parser.error("Requested region file: {0} does not exist.".format(
            args.labels))

    if not os.path.isfile(args.lut):
        parser.error("Freesurfer LUT file: {0} does not exist.".format(
            args.lut))


    # Validate that tracts can be processed
    if not validate_coordinates(args.aparc, args.tracts, nifti_compliant=True):
        parser.error("The tracts file contains points that are invalid.\n" +
                     "Use the remove_invalid_coordinates.py script to clean.")

    # Load label image
    labels_img = nib.load(args.aparc)
    full_labels = labels_img.get_data().astype('int')

    # Load fibers
    tract_format = tc.detect_format(args.tracts)
    tract = tract_format(args.tracts, args.aparc)
    
    affine = compute_affine_for_dipy_functions(args.aparc, args.tracts)

    
    # ========= processing streamlines =================
    fiberlen_range = np.asarray([args.minlen,args.maxlen])

    streamlines = [t for t in tract]
    print "Subject "+ args.sub_id + " has " + str(len(streamlines)) + " raw streamlines."

    f_streamlines = [] #filtered streamlines
    lenrecord = []
    idx = 0
    for sl in streamlines:
        # Avoid streamlines having only one point, as they crash the
        # Dipy connectivity matrix function.
        if sl.shape[0] > 1:
            flen = length(sl)
            # get fibers having length between 20mm and 200mm
            if (flen>fiberlen_range[0]) & (flen<fiberlen_range[1]):
                f_streamlines.append(sl)
                lenrecord.append(flen)
                idx = idx + 1
    
    print "Subject "+ args.sub_id + " has " + str(idx) + " streamlines with lengths between " + str(args.minlen) + " and " + str(args.maxlen) + "."
    
    # ============= process the parcellation =====================
    dilation_para = np.array([args.dilation_dist,args.dilation_windsize])

    # Compute the mapping from label name to label id
    label_id_mapping = compute_labels_map(args.lut)

    # Find which labels were requested by the user.
    requested_labels_mapping = compute_requested_labels(args.labels,
                                                        label_id_mapping)
    
    # Filter to keep only needed ones
    filtered_labels = np.zeros(full_labels.shape, dtype='int')
    for label_val in requested_labels_mapping:
        if sum(sum(sum(full_labels == label_val)))==0:
            print label_val
            print requested_labels_mapping[label_val]

        filtered_labels[full_labels == label_val] = label_val
    

    #cortex band dilation
    dilated_labels = cortexband_dilation_wm(filtered_labels,full_labels,dilation_para)

    # Reduce the range of labels to avoid a sparse matrix,
    # because the ids of labels can range from 0 to the 12000's.
    reduced_labels, labels_lut = dpu.reduce_labels(filtered_labels)
    reduced_dilated_labels, labels_lut = dpu.reduce_labels(dilated_labels)
    

    # Compute connectivity matrix and extract the fibers
    M,grouping = nconnectivity_matrix(f_streamlines, reduced_dilated_labels,fiberlen_range,args.cnpoint, affine=affine,
                                symmetric=True, return_mapping=True,
                                mapping_as_streamlines=True)

    Msize = len(M)
    CM_before_outlierremove = M[1:, 1:]
    nstream_bf = np.sum(CM_before_outlierremove)
    print args.sub_id + ' ' + str(nstream_bf) + ' streamlines in the connectivity matrix before outlier removal.'
    
    #===================== process the streamlines =============
    outlier_para = 3;
    average_thrd = 8;
    
    M_after_ourlierremove = np.zeros((Msize,Msize))
    #downsample streamlines
    cell_streamlines = []
    cell_id = []
    for i in range(1,Msize):
        for j in range(i+1, Msize):
            tmp_streamlines = grouping[i,j]
            tmp_streamlines = list(tmp_streamlines)
            #downsample
            tmp_streamlines_downsampled = [downsample(s, 100) for s in tmp_streamlines]
            #remove outliers, we need to rewrite the QuickBundle method to speed up this process 

            qb = QuickBundles(threshold=average_thrd)
            clusters = qb.cluster(tmp_streamlines_downsampled)
            outlier_clusters = clusters < outlier_para #small clusters
            nonoutlier_clusters =  clusters[np.logical_not(outlier_clusters)]
            
            tmp_nonoutlier_index = []
            for tmp_cluster in nonoutlier_clusters: 
                tmp_nonoutlier_index = tmp_nonoutlier_index + tmp_cluster.indices

            clean_streamline_downsampled = [tmp_streamlines_downsampled[ind] for ind in tmp_nonoutlier_index]
            cell_streamlines.append(clean_streamline_downsampled)
            cell_id.append([i,j])
            M_after_ourlierremove[i,j] = len(clean_streamline_downsampled)

    CM_after_ourlierremove = M_after_ourlierremove[1:,1:];
    nstream_bf = np.sum(CM_after_ourlierremove)
    print args.sub_id + ' ' + str(nstream_bf) + ' streamlines in the connectivity matrix after outlier removal.'
    
    pdb.set_trace()

    #===================== save the data =======================
    cmCountMatrix_fname = args.sub_id + "_" + args.pre + "_cm_count.mat"
    cmStreamlineMatrix_fname = args.sub_id + "_" + args.pre + "_cm_streamlines.mat"
    reduced_labels_fname = args.sub_id + "_" + args.pre + "_reduced_labels.nii.gz"
    dilated_labels_fname = args.sub_id + "_" + args.pre + "_dilated_labels.nii.gz"
    RoiInfo_fname = args.sub_id + "_" + args.pre + "_RoiInfo.mat"

    # save the count matrix
    CM = M[1:, 1:]
    sio.savemat(cmCountMatrix_fname,{'cm':CM})
    
    # save the streamline matrix
    cell_streamlines = []
    cell_id = []
    for i in range(1,Msize):
        for j in range(i+1, Msize):
            tmp_streamlines = grouping[i,j]
            tmp_streamlines = list(tmp_streamlines)
            tmp_streamlines_downsampled = [downsample(s, 100) for s in tmp_streamlines]
            cell_streamlines.append(tmp_streamlines_downsampled)
            cell_id.append([i,j])
    
    sio.savemat(cmStreamlineMatrix_fname,{'slines':cell_streamlines})
    sio.savemat(RoiInfo_fname,{'ROIinfo':cell_id})
    print args.sub_id + 'cell_streamlines.mat, ROIinfo.mat has been saved'

    filtered_labels_img = nib.Nifti1Image(filtered_labels, labels_img.get_affine(),labels_img.get_header())
    nib.save(filtered_labels_img,reduced_labels_fname)
    print args.sub_id + 'filtered labels have saved'

    dilated_labels_img = nib.Nifti1Image(dilated_labels, labels_img.get_affine(),labels_img.get_header())
    nib.save(dilated_labels_img,dilated_labels_fname)
    print args.sub_id + 'dilated labels have saved'


    nsavesl = np.sum(CM)
    print args.sub_id + str(nsavesl) + 'streamlines have been saved in the connectivity matrix.'

if __name__ == "__main__":
    main()