#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os

from dipy.tracking.streamlinespeed import length
import nibabel as nib
import numpy as np
import tractconverter as tc


DESCRIPTION = """
    Given a set of streamlines, the script computes the 
    the Tract Density Imaging map (TDI) [1], the Average Pathlength 
    Map (APM) [2] and the Connectivity Directionally-Encoded Color map 
    (C-DEC) [3].

    TDI: number of streamlines, in each voxel,
    APM: Average length of streamline passing through in each voxel,
    C-DEC: Average orientation of streamlines (vector connecting 
           streamline extremities) passing through each voxel
"""

EPILOG = """
References:
    [1] Calamante, F., Tournier, J.-D., Jackson, G. D.,& Connelly, A. (2010). 
        Track-density imaging (TDI): Super-resolution white matter imaging using whole-brain 
        track-density mapping. NeuroImage, 53(4), 1233--1243.
    [2] Pannek, K., Mathias, J. L., Bigler, E. D., Brown, G., Taylor, J. D., & Rose, S. E. (2011). 
        The average pathlength map: a diffusion MRI tractography-derived index for studying brain pathology. 
        NeuroImage, 55(1), 133--141.
    [3] Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. (2014). 
        Connectivity directionally-encoded color map: a streamline-based color mapping. 
        In Organization for Human Brain Mapping. Hamburg, Germany.
"""


def buildArgsParser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION, epilog=EPILOG)

    p._optionals.title = "Options and Parameters"
    p.add_argument('tract_file', action='store', metavar='file', type=str,
                   help="Streamlines file name. Points must be equally spaced.")
    p.add_argument('ref_file', action='store', metavar='file', type=str,
                   help="Reference file name. Voxel must have an isotropic resolution" +
                        " (nifti).")

    p.add_argument('--res', dest='res', action='store', type=float, default=1.0,
                   help='Output isotropic voxel size in mm. [%(default)s]')
    p.add_argument('-f', action='store_true', dest='overwrite',
                   help='If True, the saved files volume will be overwritten ' +
                        'if they already exist [%(default)s].')
    p.add_argument('-v', action='store_true', dest='isVerbose',
                   help='If set, produces verbose output.')
    p.add_argument('--not_all', action='store_true', dest='not_all',
                   help='If set, only saves the files specified using the ' +
                        'file flags [%(default)s].')
    
    g = p.add_argument_group(title='File flags')
    g.add_argument('--tdi', action='store', dest='tdi_file',
                   metavar='file', default='', type=str,
                   help='Output filename for TDI image.')

    g.add_argument('--apm', action='store', dest='apm_file',
                   metavar='file', default='', type=str,
                   help='Output filename for the APM image.')

    g.add_argument('--cdec', action='store', dest='cdec_file',
                   metavar='file', default='', type=str,
                   help='Output filename for the C-DEC image. ')
    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    if args.isVerbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.not_all:
        if not args.tdi_file:
            args.tdi_file = 'tdi.nii.gz'
        if not args.apm_file:
            args.apm_file = 'apm.nii.gz'
        if not args.cdec_file:
            args.cdec_file = 'cdec.nii.gz'

    arglist = [args.tdi_file, args.apm_file, args.cdec_file]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')
    for out in arglist:
        if os.path.isfile(out):
            if args.overwrite:
                logging.info('Overwriting "{0}".'.format(out))
            else:
                parser.error('"{0}" already exists! Use -f to overwrite it.'.format(out))

    ref = nib.load(args.ref_file)
    ref_res = ref.get_header()['pixdim'][1]
    up_factor = ref_res / args.res
    data_shape = np.array(ref.shape) * up_factor
    data_shape = list(data_shape.astype('int32'))

    logging.info("Reference resolution: " + str(ref_res))
    logging.info("Reference shape: " + str(ref.shape))
    logging.info("Target resolution: " + str(args.res))
    logging.info("Target shape: " + str(data_shape))

    cdec_map = np.zeros(data_shape + [3], dtype='float32')
    tdi_map = np.zeros(data_shape, dtype='float32')
    apm_map = np.zeros(data_shape, dtype='float32')

    tract_format = tc.detect_format(args.tract_file)
    tract = tract_format(args.tract_file)
    streamlines = [i for i in tract]
    streamlines_np = np.array(streamlines, dtype=np.object)

    for i, streamline in enumerate(streamlines_np):
        if not i % 10000:
            logging.info(str(i) + "/" + str(streamlines_np.shape[0]))

        streamline_length = length(streamline)
        dec_vec = np.array(streamline[0] - streamline[-1])
        dec_vec_norm = np.linalg.norm(dec_vec)
        if dec_vec_norm > 0:
            dec_vec = np.abs(dec_vec / dec_vec_norm)
        else:
            dec_vec[0] = dec_vec[1] = dec_vec[2] = 0

        for point in streamline:
            pos = point / args.res
            ind = tuple(pos.astype('int32'))
            if (ind[0] >= 0 and ind[0] < data_shape[0] and
                    ind[1] >= 0 and ind[1] < data_shape[1] and
                    ind[2] >= 0 and ind[2] < data_shape[2]):
                tdi_map[ind] += 1
                apm_map[ind] += streamline_length
                cdec_map[ind] += dec_vec

    # devide the sum of streamline length by the streamline density
    apm_map /= tdi_map

    # normalise the cdec map
    cdec_norm = np.sqrt((cdec_map * cdec_map).sum(axis=3))
    cdec_map = cdec_map / cdec_norm.reshape(list(cdec_norm.shape) + [1]) * 255

    affine = ref.get_affine()
    affine[0][0] = affine[1][1] = affine[2][2] = args.res

    if args.tdi_file:
        tdi_img = nib.Nifti1Image(tdi_map, affine)
        tdi_img.get_header().set_zooms([args.res, args.res, args.res])
        tdi_img.get_header().set_qform(ref.get_header().get_qform())
        tdi_img.get_header().set_sform(ref.get_header().get_sform())
        tdi_img.to_filename(args.tdi_file)

    if args.apm_file:
        apm_img = nib.Nifti1Image(apm_map, affine)
        apm_img.get_header().set_zooms([args.res, args.res, args.res])
        apm_img.get_header().set_qform(ref.get_header().get_qform())
        apm_img.get_header().set_sform(ref.get_header().get_sform())
        apm_img.to_filename(args.apm_file)

    if args.cdec_file:
        cdec_img = nib.Nifti1Image(cdec_map.astype('uint8'), affine)
        cdec_img.get_header().set_zooms([args.res, args.res, args.res, 1])
        cdec_img.get_header().set_qform(ref.get_header().get_qform())
        cdec_img.get_header().set_sform(ref.get_header().get_sform())
        cdec_img.to_filename(args.cdec_file)


if __name__ == "__main__":
    main()
