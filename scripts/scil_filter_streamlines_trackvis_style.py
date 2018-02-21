#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import xml.etree.ElementTree as ET

from dipy.core.geometry import dist_to_corner
from dipy.tracking.utils import target, near_roi

import nibabel as nb
import numpy as np

from scilpy.io.streamlines import ichunk


def _filter_from_roi_trackvis_style(tracts_fname, roi_fname, out_tracts_fname,
                                    include, mode, tol):
    tracts, hdr = nb.trackvis.read(tracts_fname, as_generator=True,
                                   points_space='voxel')
    img = nb.load(roi_fname)

    aff = np.eye(4)
    filtered_strl = []

    mask = img.get_data()

    for chunk in ichunk(tracts, 100000):
        logging.debug("Starting new loop")

        # Need to "unshift" because Dipy shifts them back to fit in voxel grid.
        # However, to mimic the way TrackVis filters, we need to "unshift" it.
        # TODO check no negative coordinates
        strls = [s[0] - 0.5 for s in chunk]

        # Interpretation from @mdesco code:
        # target is used for this case because we want to support
        # include=False.
        if mode == 'any_part' and tol == 0:
            target_strl_gen = target(strls, mask, aff, include=include)
            target_strl = list(target_strl_gen)
            if len(target_strl) > 0:
                filtered_strl.extend(target_strl)
        else:
            corner_dist = dist_to_corner(aff)

            # Tolerance is considered to be in mm. Rescale wrt voxel size.
            tol = tol / min(img.get_header().get_zooms())

            # Check this to avoid Dipy's warning and scare users.
            if tol < corner_dist:
                logging.debug("Setting tolerance to minimal distance to corner")
                tol = corner_dist

            if mode == 'any_part':
                strls_ind = near_roi(strls, mask, aff, tol=tol, mode='any')
            else:
                strls_ind = near_roi(strls, mask, aff, tol=tol, mode=mode)

            for i in np.where(strls_ind)[0]:
                filtered_strl.append(strls[i])

    logging.info("Original nb: {0}, kept: {1}".format(hdr['n_count'],
                                                      len(filtered_strl)))

    # Remove the unshift
    final_strls = [(s + 0.5, None, None) for s in filtered_strl]

    new_hdr = np.copy(hdr)
    new_hdr['n_count'] = len(filtered_strl)
    nb.trackvis.write(out_tracts_fname, final_strls, new_hdr,
                      points_space="voxel")


def _filter_from_sphere_trackvis_style(tracts_fname, sphere_center, sphere_size,
                                       out_tracts_fname, include):
    tracts, hdr = nb.trackvis.read(tracts_fname, as_generator=True,
                                   points_space='voxel')

    filtered_strl = []

    def _is_included(norms, size):
        return np.any(norms <= size)

    def _is_excluded(norms, size):
        return np.all(norms > size)

    if include:
        filter_func = _is_included
    else:
        filter_func = _is_excluded

    for chunk in ichunk(tracts, 100000):
        logging.debug("Starting new loop")

        # TODO check no negative coordinates
        strls = [s[0] for s in chunk]
        vects = [s - sphere_center for s in strls]
        norms = [np.linalg.norm(d, axis=1) for d in vects]

        # TODO add either_end and both_end
        for strl_idx, strl_points_norms in enumerate(norms):
            # if np.any(strl_points_norms <= sphere_size):
            if filter_func(strl_points_norms, sphere_size):
                filtered_strl.append((strls[strl_idx], None, None))

        logging.debug("Filtered streamlines now has {0} elements".format(len(filtered_strl)))

    new_hdr = np.copy(hdr)
    new_hdr['n_count'] = len(filtered_strl)
    nb.trackvis.write(out_tracts_fname, filtered_strl, new_hdr,
                      points_space="voxel")


def _find_roi_sphere_element_by_name(rois_element, sphere_name):
    child_elements = rois_element.findall('ROI')

    for roi_el in child_elements:
        if roi_el.attrib['name'] == sphere_name and\
           roi_el.attrib['type'] == 'Sphere':
            return roi_el

    return None


def _read_sphere_info_from_scene(scene_filename, sphere_name):
    # The trackvis scene format is not valid XML (surprise, surprise...)
    # because there are 2 "root" elements.
    # We need to fake it by reading the file ourselves and adding a fake root
    with open(scene_filename, 'rb') as scene_file:
        scene_str = '<fake_root>' + scene_file.read() + '</fake_root>'

    # We need to filter out the P## element that Trackvis likes to add to the
    # scene.
    coord_start = scene_str.find('<Coordinate>')
    coord_end = scene_str.find('</Coordinate>')
    coord_end += len('</Coordinate>')
    scene_str = scene_str.replace(scene_str[coord_start:coord_end], '')

    root = ET.fromstring(scene_str)

    scene_el = root.find('Scene')

    dim_el = scene_el.find('Dimension')

    x_dim = int(dim_el.attrib.get('x'))
    y_dim = int(dim_el.attrib.get('y'))

    voxel_order = scene_el.find('VoxelOrder').attrib['current']

    if voxel_order != 'LPS':
        raise IOError('Scene file voxel order was not LPS, unsupported.')

    rois_el = scene_el.find('ROIs')

    sphere_el = _find_roi_sphere_element_by_name(rois_el, sphere_name)

    if sphere_el is None:
        raise IOError('Scene file did not contain a sphere called {0}'.format(sphere_name))

    center_el = sphere_el.find('Center')
    center_x = float(center_el.attrib['x'])
    center_y = float(center_el.attrib['y'])
    center_z = float(center_el.attrib['z'])

    size_el = sphere_el.find('Radius')
    sphere_size = float(size_el.attrib['value'])

    # Need to flip the X and Y coordinates since they were in LPS
    center_x = x_dim - 1 - center_x
    center_y = y_dim - 1 - center_y

    return np.array([center_x, center_y, center_z]), sphere_size


def _buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='This script mimics Trackvis filtering behavior, so you ' +
                    'can call it from the command line.\n\n' +
                    'You can use it to filter using 2 different sources:\n'
                    '- an image-based ROI, defining a mask\n'
                    '- a sphere-based ROI, defining a spherical ROI in world '
                    'space.\n\n'
                    'Note that only one mode can be chosen at once.\n\n'
                    'To filter using a sphere, simply give the path to the '
                    'scene file,\nas well as the sphere name. This avoids '
                    'having multiple arguments\nfor the space (LPS, RAS) of '
                    'sphere definition. Note that this\ncurrently only supports '
                    'LPS.\n\n'
                    'You can use the --not option to exclude tracts going '
                    'through your ROI or sphere.\n\n'
                    'IMPORTANT: this doesn\'t currently correctly handle '
                    'compressed streamlines.')

    p.add_argument('tracts_file', action='store',
                   metavar='TRACTS', type=str,
                   help='path of tracts file in trk format, coming from PFT tracking')

    p.add_argument('out_tracts', action='store', metavar='TRACTS', type=str,
                   help='output tracts file')

    p.add_argument('--roi_file', action='store',
                   metavar='ROI', type=str,
                   help='filter using the ROI in the provided nifti image')

    p.add_argument('--sphere', action='store',
                   metavar='SPHERE_INFO', type=str, nargs=2,
                   help='filter using a sphere. Need to provide the scene file ' +
                        'and the name of the sphere, like: sphere.scene VL-L')

    p.add_argument('--not', action='store_false', dest='include',
                   help='instead of keeping tracts going through ROI, exclude them.')

    p.add_argument('--mode', action='store',
                   dest='mode', metavar='MODE_TYPE',
                   choices=['any_part', 'either_end', 'both_end'],
                   help="'any_part', 'either_end', or 'both_end' points are " +
                        "within the ROI. Does not apply to --not and --sphere" +
                        " options. (Default 'any_part').")

    p.add_argument('--tol', action='store',
                   dest='tol', metavar='mm', type=float,
                   help='Tolerance in mm from the ROI for the --mode option. ' +
                        'Does not apply to --not and --sphere options. ' +
                        '(Default: 0.0).')

    p.add_argument('-f', action='store_true', dest='force_output',
                   help='Force (overwrite output file). [%(default)s]')
    p.add_argument('--debug', action='store_true', dest='debug',
                   help='use debugging logging')

    return p


def main():
    parser = _buildArgsParser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isfile(args.tracts_file):
        parser.error('"{0}" must be a file!'.format(args.tracts_file))

    # TODO validate this is a trackvis file

    if os.path.isfile(args.out_tracts) and not args.force_output:
        parser.error("Output file ('{0}') already exists. ".format(args.out_tracts) +
                     "Remove or use -f.")

    if args.roi_file and args.sphere:
        parser.error('You can only use ROI filtering or Sphere filtering at ' +
                     'one time.')

    if args.roi_file and not os.path.isfile(args.roi_file):
        parser.error('"{0}" must be a file!'.format(args.roi_file))

    if args.sphere and (args.mode or args.tol):
        parser.error("'--mode' and '--tol' arguments not supported for " +
                     "sphere mode.")

    # Check if it was set
    if not args.include and (args.mode or args.tol):
        parser.error("'--mode' and '--tol' arguments not supported for " +
                     "'--not' mode.")

    # Set defaults if not set
    if args.roi_file and not args.mode:
        args.mode = 'any_part'

    if args.roi_file and not args.tol:
        args.tol = 0.0

    if args.roi_file:
        _filter_from_roi_trackvis_style(args.tracts_file, args.roi_file,
                                        args.out_tracts, args.include, args.mode,
                                        args.tol)
    elif args.sphere:
        sphere_center, sphere_size = _read_sphere_info_from_scene(args.sphere[0],
                                                                  args.sphere[1])
        _filter_from_sphere_trackvis_style(args.tracts_file, sphere_center,
                                           sphere_size, args.out_tracts,
                                           args.include)


if __name__ == "__main__":
    main()
