#! /usr/bin/env python

from __future__ import division, print_function

import dicom
import os
import argparse
import logging

import numpy as np
import nibabel as nib


DESCRIPTION = """
    Convert a Philips 3T enhanced DICOM output from the scanner to Nifti files.
    Echo spacing parameters for topup and gradient tables will be outputted in text files.
    
    WARNING: Note that this script is only made to support diffusion data
    acquired on the Sherbrooke Philips 3T when dicom data is saved with
    "enhanced dicom". It is not meant to support anything else!!!

    WARNING: This is a highly experimental script. Use at your own risks.
"""

def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('out_dir', action='store', type=str,
                    help='Output directory where Nifti and text files are written.')

    mg = p.add_mutually_exclusive_group(required=True)
    mg.add_argument('--dicom_dir', action='store', dest='dicom_dir', metavar='DIR', type=str,
                    help='dicom directory. name of the directory ' +
                         'containing the dicom files.')
    mg.add_argument('--dicom_files', dest='dicom_file_list', action='store',
                    metavar=' ', type=str, nargs='+',
                    help='dicom files. list of the names of the ' +
                         'dicom file.')
    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='If set, the saved files will be overwritten ' +
                        'if they already exist. (Default: False)')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.warning("This script is highly experimental. Use at your own risks.")

    if not os.path.isdir(args.out_dir):
        parser.error('"{0}" does not exist! Please provide a valid output directory.'.format(args.out_dir))

    # Load all dicom files
    if args.dicom_dir:
        dicom_files = [os.path.join(args.dicom_dir, f) 
                       for f in sorted(os.listdir(args.dicom_dir))]
    elif args.dicom_file_list:
        dicom_files = [f for f in args.dicom_file_list]


    for f in dicom_files:
        # if file is a txt file bval bvec, we skip
        if "txt" in f or "bval" in f or "bvec" in f or "nii" in f:
            continue

        # if file does not exist, we stripped .nii.gz or .txt, it's not a dicom anyway
        if not os.path.isfile(f) or "IM" not in f:
            continue

        # if file exists, we don't reprocess it
        filename = os.path.join(args.out_dir, os.path.basename(f))
        if os.path.isfile(os.path.join(filename, ".nii.gz")) :
            if args.overwrite:
                logging.info('Overwriting "{0}".'.format( os.path.join(filename, ".nii.gz")))
            else:
                parser.error('"{0}" already exists! Use -f to overwrite it.'.format(os.path.join(filename,
                                                                                                 ".nii.gz")))

        dicom_file = dicom.read_file(f)
        logging.info('Loaded dicom file is "{0}"'.format(f))

        # Private tags for the 3T
        #  http://www.na-mic.org/Wiki/index.php/NAMIC_Wiki:DTI:DICOM_for_DWI_and_DTI#Private_vendor:_Philips
        #    (2001,1003) : B_value
        #    (2001,1004) : Diffusion direction
        #
        # for SS values Private_2005_140f, Private_2005_100e
        # for SI values Private_2005_140f, Private_2005_100d

        # List of all pydicom known tags
        # https://github.com/darcymason/pydicom/blob/master/source/dicom/_dicom_dict.py

        num_slices = dicom_file[0x2001, 0x1018].value
        dicom_data = np.empty(dicom_file.pixel_array.shape[::-1])
        for i in range(dicom_data.shape[-1]):
            dicom_data[..., i] = dicom_file.pixel_array[i].T

        shape = dicom_data.shape

        # My test file is a 3D array, so let's hope that's standard since
        # guessing the order of slices is (one of) the reason(s) why parsing dicom is hell
        if len(shape) != 3:
            raise ValueError("dicom file is not stored as a 2D array stack, and I don't know how to process it",
                             shape)
        print((shape[0], shape[1], num_slices, np.int(shape[2]/num_slices)))
        reordered_data = np.zeros((shape[0], shape[1], num_slices, 
                                   np.int(shape[2]/num_slices)), dtype=np.float32)
        try:
            acquisition_dimension = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157].value
        except AttributeError:
            print("A required tag was missing, if this is a classical dicom, please use mrtrix instead")
            continue

        # 3D only, it's a noise map / reversed b0
        if len(acquisition_dimension) == 3:

            reordered_data = np.squeeze(reordered_data)

            for idx in range(reordered_data.shape[-1]):

                # Scaling proportional to the measured MRI signal
                SS = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100e].value
                SI = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100d].value

                SliceDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][1]

                if len(reordered_data.shape) == 3 : # need to add this. Somehow, passed the above if
#                    print(reordered_data.shape, dicom_data.shape, len(acquisition_dimension),
#                          len(reordered_data.shape))
                    reordered_data[..., SliceDiff-1] =  np.array((dicom_data[..., idx] - SI) / SS, dtype=np.float32)
                    
                # Noise map/b0 acquired by themselves are actually two acquisitions averaged by the scanner,
                # so we must sqrt2 them
                # reordered_data /= np.sqrt(2)

            # dummy bval/bvec to not crash the saving part
            bvals = np.array([0.])
            bvecs = np.zeros((3, 1), dtype=np.float32)

        # 4D is present, it's a set of dwi
        elif len(acquisition_dimension) == 4:

            SliceDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][1]
            prev_shell     = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][2]
            IndexDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][3]
            idx = 0
            idx_used = np.zeros(reordered_data.shape[2:], dtype=np.bool)

            # Get data
            for t in range(reordered_data.shape[-1]):
                for z in range(num_slices):

                    SliceDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][1]
                    Shell      = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][2]
                    IndexDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][3]

                    # b0 only acquisition can have a gradient index higher than the dimension of the data for some reason
                    if IndexDiff > idx_used.shape[-1]:
                        IndexDiff = idx_used.shape[-1]

                    # Current index is used, so find an empty one
                    if idx_used[SliceDiff-1, IndexDiff-1]:
                        IndexDiff = np.asscalar(np.argwhere(idx_used[SliceDiff-1] == 0)[0]) + 1

                    SliceDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][1]
                    prev_shell     = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][2]
                    # IndexDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][3]

                    # Scaling proportional to the measured MRI signal
                    # this line is actually the whole point of the parser
                    SS = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100e].value
                    SI = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100d].value

                    # Dicom stores the data starting from 1 for indexing, while python starts from 0
                    reordered_data[..., SliceDiff-1, IndexDiff-1] =  np.array((dicom_data[..., idx] - SI) / SS, dtype=np.float32)
                    idx_used[SliceDiff-1, IndexDiff-1] = True
                    idx += 1


        # 5D is (multishell) data with different image types inside
        # SliceType is the type of 3D image acquired, i.e. 0 for magnitude image, 3 for phase image
        elif len(acquisition_dimension) == 5:

            SliceDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][1]
            SliceType_prev = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][2]
            Shell_prev     = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][3]
            IndexDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[0][0x0020, 0x9111][0][0x0020, 0x9157][4]
            idx = 0
            idx_used = np.zeros(reordered_data.shape[2:], dtype=np.bool)

            # Get data
            for t in range(reordered_data.shape[-1]):
                for z in range(num_slices):

                    SliceDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][1]
                    SliceType  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][2]
                    Shell      = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][3]
                    IndexDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][4]

                    # Acquisition can have a gradient index higher than the dimension of the data for some reason

                    # Current index is used by previous magnitude image, so find an empty one
                    if idx_used[SliceDiff-1, IndexDiff-1] and SliceType != 3:
                        IndexDiff = np.asscalar(np.argwhere(idx_used[SliceDiff-1] == 0)[0]) + 1

                    #print(IndexDiff-1,idx,reordered_data.shape,SliceDiff)
                    # We have both magnitude and phase in Marco's data,
                    # so put all magnitude images as the first half then all phase images as the last half.
                    if SliceType == 3:
                        IndexDiff += reordered_data.shape[-1] / 2
                    # print(IndexDiff-1, idx, IndexDiff_prev)
                    # if IndexDiff-1 > 290:
                    #print(IndexDiff-1,idx,reordered_data.shape,SliceDiff)

                    #   IndexDiff=290
                    # Scaling proportional to the measured MRI signal
                    SS = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100e].value
                    SI = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x2005, 0x140f][0][0x2005, 0x100d].value

                    SliceDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][1]
                    SliceType_prev = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][2]
                    Shell_prev     = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][3]
                    IndexDiff_prev = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][4]
                    # IndexDiff=0
                    # Dicom stores the data starting from 1 for indexing, while python starts from 0
                    # print(reordered_data.shape, reordered_data[..., SliceDiff-1, IndexDiff-1].shape, SliceDiff-1, IndexDiff-1)
                    reordered_data[..., SliceDiff-1, IndexDiff-1] =  np.array((dicom_data[..., idx] - SI) / SS, dtype=np.float32)
                    idx_used[SliceDiff-1, IndexDiff-1] = True
                    idx +=1

        else:
            logging.info('I cannot guess if your file is a set of dwis, ' + \
                             'a b0/noise map or multishell data. Skipping it ...')
            continue

        # Flip data 180 in Z
        for i in range(reordered_data.shape[2]):
            reordered_data[:, :, i] = np.rot90(reordered_data[:, :, i], k=2)

        # Get bvals/bvecs
        # bvec (0018, 9117) (0018, 9076) (0018, 9089) Diffusion Gradient Orientation
        # bval (0018, 9117) (0018, 9087) Diffusion b-value
        if len(acquisition_dimension) > 3:

            bvals = np.zeros((1, reordered_data.shape[-1]), dtype=np.float32)
            bvecs = np.zeros((3, reordered_data.shape[-1]), dtype=np.float32)

            # no bval, but it's 4D, weird so we skip it
            if [0x0018, 0x9117] not in dicom_file.PerFrameFunctionalGroupsSequence[0]:
                break

            for idx in range(reordered_data.shape[-1]):

                # position in the array of the current bval/bvec, indexing starts from 1 so we -1 later
                bval = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0018, 0x9117][0][0x0018, 0x9087].value
                IndexDiff  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][-1]

            # b0 only acquisition can have a gradient index higher than the dimension of the data for some reason
                if IndexDiff > idx_used.shape[-1]:
                    IndexDiff = idx_used.shape[-1]

            #print(bval)
            # 5D multishell, we put all the magnitude then the phase images, so we need to account for that
                if len(acquisition_dimension) == 5:

                    SliceType  = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0020, 0x9111][0][0x0020, 0x9157][2]

                # Current index is used by previous magnitude image, so find an empty one
                    if idx_used[SliceDiff-1, IndexDiff-1] and SliceType != 3:
                        IndexDiff = np.asscalar(np.argwhere(idx_used[SliceDiff-1] == 0)[0]) + 1

                # We have both magnitude and phase in Marco's data,
                # so put all magnitude images as the first half then all phase images as the last half.
                    if SliceType == 3:
                        IndexDiff += reordered_data.shape[-1] / 2

            # check if the bvec tag exists
                if [0x0018, 0x9076] in dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0018, 0x9117][0]:
                    bvec = dicom_file.PerFrameFunctionalGroupsSequence[idx][0x0018, 0x9117][0][0x0018, 0x9076][0][0x0018, 0x9089].value
            # no bvec tag, so it's a b0
                else:
                    bvec = [0, 0, 0]

            # if bval < 1, it's a b0, so put bvec to 0 also
                if bval < 1:
                    bvals[:, IndexDiff-1] = 0
                    bvecs[:, IndexDiff-1] = [0., 0., 0]
                else:
                    bvals[:, IndexDiff-1] = bval
                    bvecs[:, IndexDiff-1] = bvec

    # Flip bvec 180 in Z
        for i in range(bvecs.shape[1]):
            bvecs[2, i] *= -1


        # In[276]:

        # Some tags
        # https://github.com/INCF/NeuroimagingMetadata/wiki/PhilipsAchieva6

        # EPI unwarping for FSL topup
        # ETL : EPI train length = dicom_file.GradientEchoTrainLength
        # ES : Echo Spacing
        # time between centres of consecutive echoes = (ES * (ETL-1))
        # WFS_pixel : Water fat shift in pixel = tag (2001, 1022)
        # wfs_hz = 434.214 hz: water fat shift in hertz, for a 3T scanner at gamma = 42.57 MHz/T and water-fat shift as 3.4 ppm
        # ES = 1000 * WFS_pixel / (wfs_hz * (ETL + 1))

        ETL = dicom_file.GradientEchoTrainLength
        WFS_pixel = dicom_file[0x2001, 0x1022].value
        wfs_hz = 434.214
        ES = WFS_pixel / (wfs_hz * (ETL + 1))

        # Formula is according to a secret pdf from Philips and is different from fsl wiki on purpose
        # http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP/TopupUsersGuide#Configuration_files
        dwell_time = ES * (ETL-1)

        # If multiple b0s, we suppose they are acquired in the same direction
        # If only one b0, we reversed the phase encode sign
        if len(reordered_data.shape) == 4:
            if bvals[-1].all() == 0 :
                continue;
            else :
                b0s = reordered_data[..., bvals[-1] == 0]
                n_b0s = b0s.shape[-1]
                phase_dir = -1
        else:
            b0s = reordered_data
            n_b0s = 1
            phase_dir = 1

        # Add all the b0s to topup and save them
        topup = "# We hardcoded the phase encode axis in y for now, cause it's easier, so double check that this is indeed the case...\n"
        for _ in range(n_b0s):
            topup += "0 " + str(phase_dir) +" 0 " + str(dwell_time) + "\n"

        logging.info('topup parameters file and b0s are saved, but you need to '+\
                         'concatenate them appropriately with the reversed phase encode b0!')

        filename = args.out_dir + os.path.basename(f)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(reordered_data, affine), filename + '.nii.gz')
        nib.save(nib.Nifti1Image(b0s, affine), filename + '_b0s.nii.gz')

        np.savetxt(filename + '.bval', bvals, fmt="%i")
        np.savetxt(filename + '.bvec', bvecs, fmt="%8f")

        topuptxt = open(filename + '_topup.txt', "w")
        topuptxt.write(topup)
        topuptxt.close()

if __name__ == "__main__":
    main()
