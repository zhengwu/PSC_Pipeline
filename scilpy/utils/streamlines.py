#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from functools import reduce
import itertools
import copy
import logging
import six

from dipy.segment.clustering import QuickBundles
from dipy.tracking import metrics as tm
from dipy.tracking.streamline import transform_streamlines
import nibabel as nb
from nibabel.streamlines import Tractogram
from nibabel.streamlines.header import Field
from nibabel.streamlines.trk import (get_affine_rasmm_to_trackvis,
                                     get_affine_trackvis_to_rasmm)
import numpy as np
from scipy import ndimage
import tractconverter as tc
from tractconverter.formats.header import Header as tract_header


MIN_NB_POINTS = 10
KEY_INDEX = np.concatenate((range(5), range(-1, -6, -1)))


def load_in_voxel_space(streamlines, anat, raise_on_empty=True):
    """
    Load streamlines in voxel space, corner aligned

    :param streamlines: path or nibabel object
    :param anat: path or nibabel image
    :return: NiBabel tractogram with streamlines loaded in voxel space
    """
    if isinstance(streamlines, six.string_types):
        nib_object = nb.streamlines.load(streamlines)
    else:
        nib_object = streamlines

    affine_to_voxmm = get_affine_rasmm_to_trackvis(nib_object.header)

    if nib_object.header[Field.NB_STREAMLINES] == 0 and raise_on_empty:
        raise Exception("The bundle contains no streamline")

    tractogram = nib_object.tractogram
    tractogram.apply_affine(affine_to_voxmm)
    streamlines = tractogram.streamlines

    if isinstance(anat, six.string_types):
        anat = nb.load(anat)

    spacing = anat.header['pixdim'][1:4]
    streamlines._data /= spacing
    return streamlines


def save_from_voxel_space(streamlines, anat, ref_tracts, out_name):
    if isinstance(ref_tracts, six.string_types):
        nib_object = nb.streamlines.load(ref_tracts, lazy_load=True)
    else:
        nib_object = ref_tracts

    if isinstance(anat, six.string_types):
        anat = nb.load(anat)

    affine_to_rasmm = get_affine_trackvis_to_rasmm(nib_object.header)

    tracto = Tractogram(streamlines=streamlines,
                        affine_to_rasmm=affine_to_rasmm)

    spacing = anat.header['pixdim'][1:4]
    tracto.streamlines._data *= spacing

    nb.streamlines.save(tracto, out_name, header=nib_object.header)


def validate_coordinates(anat, streamlines, nifti_compliant=True):
    # Check if all points in the tracts are inside the image volume.
    ref_img = nb.load(anat)
    voxel_dim = ref_img.get_header()['pixdim'][1:4]

    if nifti_compliant:
        shift_factor = voxel_dim * 0.5
    else:
        shift_factor = voxel_dim * 0.0

    tract_file = streamlines
    if isinstance(streamlines, six.string_types):
        tc_format = tc.detect_format(streamlines)
        tract_file = tc_format(streamlines, anatFile=anat)

    # TODO what check to do for .vtk?
    if isinstance(tract_file, tc.formats.tck.TCK) \
       or isinstance(tract_file, tc.formats.trk.TRK):
        for s in tract_file:
            strl = np.array(s + shift_factor)
            if np.any(strl < 0):
                return False
    else:
        raise TypeError("This function currently only supports TCK and TRK.")

    return True


def get_tract_count(streamlines):
    if isinstance(streamlines, six.string_types):
        tc_format = tc.detect_format(streamlines)
        tract_file = tc_format(streamlines)
        tract_count = tract_file.hdr[tract_header.NB_FIBERS]
    elif isinstance(streamlines, list):
        tract_count = len(streamlines)
    # Need to do it like this since the is no parent class in the formats.
    elif isinstance(streamlines, tc.formats.tck.TCK) \
        or isinstance(streamlines, tc.formats.trk.TRK) \
        or isinstance(streamlines, tc.formats.vtk.VTK):
        tract_count = streamlines.hdr[tract_header.NB_FIBERS]

    return tract_count


def remove_loops_and_sharp_turns(streamlines, use_qb, max_angle, qb_threshold,
                                 logger=None):
    """
    Remove loops and sharp turns from a list of streamlines.

    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    use_qb: bool
        Set to True if the additional QuickBundles pass is done.
        This will help remove sharp turns. Should only be used on
        bundled streamlines, not on whole-brain tractograms.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.
    qb_threshold: float
        Quickbundles distance threshold, only used if use_qb is True.
    logger: logging object, optional
        Logger to use.

    Returns
    -------
    A tuple containing
        list of ndarray: the clean streamlines
        list of ndarray: the list of removed streamlines, if any
    """
    if logger is None:
        logger = logging.getLogger()

    loops = []
    streamlines_clean = []
    for s in streamlines:
        if tm.winding(s) >= max_angle:
            loops.append(s)
        else:
            streamlines_clean.append(s)

    if use_qb:
        if len(streamlines_clean) > 1:
            streamlines = streamlines_clean
            curvature = []
            streamlines_clean = []

            qb = QuickBundles(threshold=qb_threshold)
            clusters = qb.cluster(streamlines)

            for cc in clusters.centroids:
                curvature.append(tm.mean_curvature(cc))
            mean_curvature = sum(curvature)/len(curvature)

            for i in xrange(len(clusters.centroids)):
                if tm.mean_curvature(clusters.centroids[i]) > mean_curvature:
                    for indice in clusters[i].indices:
                        loops.append(streamlines[indice])
                else:
                    for indice in clusters[i].indices:
                        streamlines_clean.append(streamlines[indice])
        else:
            logger.warning("Impossible to use the use_qb option because " +
                           "not more than one streamline left from the\n" +
                           "input file.")

    return streamlines_clean, loops


def substract_streamlines(streamlines, streamlines_to_remove, logger=None):
    """Subtracts streamlines from a list

    Removes the streamlines of 'streamlines' that are in
    'streamlines_to_remove'. Every streamline in 'streamlines_to_remove' must
    have an exact match (identical points) in 'streamlines'.

    Args:
        streamlines (list of ndarray) : The list of streamlines from which
            we remove streamlines.
        streamlines_to_remove (list of ndarray) : The list of streamlines
            to be removed. Every element of this list must be present in the
            streamlines.
        logger (logging object) : Logger to use.

    """

    if logger is None:
        logger = logging.getLogger()
    logger.warn('The function substract_streamlines is deprecated. Use '
                'perform_streamlines_operation instead.')

    # Hash all streamlines. We need to make the data unwriteable to use it
    # as a key.
    logger.info('Building streamline dict ...')
    streamlines_dict = {}
    for i, streamline in enumerate(streamlines):

        # Use just a few data points as hash key. I could use all the data of
        # the streamlines, but then the complexity grows with the number of
        # points.
        if len(streamline) < MIN_NB_POINTS:
            key = streamline
        else:
            key = streamline[KEY_INDEX]
        key.flags.writeable = False
        streamlines_dict[key.data] = i

    # Find the indices of the streamlines to remove.
    logger.info('Finding streamlines to remove ...')
    indices_to_remove = set()
    for streamline_to_remove in streamlines_to_remove:
        if len(streamline_to_remove) < MIN_NB_POINTS:
            key = streamline_to_remove
        else:
            key = streamline_to_remove[KEY_INDEX]
        key.flags.writeable = False

        if key.data in streamlines_dict:
            indices_to_remove.add(streamlines_dict[key.data])
        else:
            logger.warning('Could not find an exact match for a ' +
                            'streamline. Ignoring it.')

    # Remove the streamlines.
    logger.info(
        'Removing {0} streamlines ...'.format(len(indices_to_remove)))
    for i in reversed(sorted(list(indices_to_remove))):
        streamlines.pop(i)


def get_streamline_key(streamline, precision=None):

    # Use just a few data points as hash key. I could use all the data of
    # the streamlines, but then the complexity grows with the number of
    # points.
    if len(streamline) < MIN_NB_POINTS:
        key = streamline.copy()
    else:
        key = streamline[KEY_INDEX].copy()

    if precision is not None:
        key = np.round(key, precision)

    key.flags.writeable = False

    return key.data


def hash_streamlines(streamlines, start_index=0, precision=None):
    """Produces a dict from streamlines

    Produces a dict from streamlines by using the points as keys and the
    indices of the streamlines as values.

    Args:
        streamlines (list of ndarray) : The list of streamlines used to
            produce the dict.
        start_index (int, optional) : The index of the first streamline.
            0 by default.
        precision (int, optional) : The number of decimals to keep when
            hashing the points of the streamlines. Allows a soft
            comparison of streamlines. If None, no rounding is performed.

    Returns:
        A dict where the keys are streamline points and the values are
        indices starting at start_index.

    """

    keys = [get_streamline_key(s, precision) for s in streamlines]
    return {k: i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k in right}


def subtraction(left, right):
    """Subtraction of two streamlines dict (see hash_streamlines)"""
    return {k: v for k, v in left.items() if k not in right}


def perform_streamlines_operation(operation, streamlines, precision=None):
    """Peforms an operation on a list of list of streamlines

    Given a list of list of streamlines, this function applies the operation
    to the first two lists of streamlines. The result in then used recursively
    with the third, fourth, etc. lists of streamlines.

    A valid operation is any function that takes two streamlines dict as input
    and produces a new streamlines dict (see hash_streamlines). Union,
    subtraction, and intersection are valid examples of operations.

    Args:
        operation (callable) : A callable that takes two streamlines dicts as
            inputs and preduces a new streamline dict.
        streamlines (list of list of streamlines) : The streamlines used in
            the operation.
        precision (int, optional) : The number of decimals to keep when
            hashing the points of the streamlines. Allows a soft
            comparison of streamlines. If None, no rounding is performed.

    Returns:
        The streamlines obtained after performing the operation on all the
            input streamlines.
        The indices of the streamlines that are used in the output.

    """

    # Hash the streamlines using the desired precision.
    indices = np.cumsum([0] + [len(s) for s in streamlines[:-1]])
    hashes = [hash_streamlines(s, i, precision) for
              s, i in zip(streamlines, indices)]

    # Perform the operation on the hashes and get the output streamlines.
    to_keep = reduce(operation, hashes)
    all_streamlines = list(itertools.chain(*streamlines))
    indices = sorted(to_keep.values())
    streamlines = [all_streamlines[i] for i in indices]
    return streamlines, indices


def union(left, right):
    """Union of two streamlines dict (see hash_streamlines)"""

    # In python 3 : return {**left, **right}
    result = left.copy()
    result.update(right)
    return result


def modify_tractogram_header_using_anat_header(in_header, ref_img):
    new_header = copy.deepcopy(in_header)
    new_header[nb.streamlines.Field.VOXEL_SIZES] = tuple(ref_img.header.
                                                         get_zooms())[:3]
    new_header[nb.streamlines.Field.DIMENSIONS] = tuple(ref_img.get_shape())[:3]
    new_header[nb.streamlines.Field.VOXEL_TO_RASMM] = ref_img.affine

    new_header[nb.streamlines.Field.VOXEL_ORDER] = ''.join(
        nb.aff2axcodes(new_header[nb.streamlines.Field.VOXEL_TO_RASMM]))

    return new_header


def warp_tractogram(streamlines, transfo, deformation_data, source):
    if source == 'ants':
        flip = [-1, -1, 1]
    elif source == 'dipy':
        flip = [1, 1, 1]

    # Because of duplication, an iteration over chunks of points is necessary
    # for a big dataset (especially if not compressed)
    nb_points = len(streamlines._data)
    current_position = 0
    chunk_size = 1000000
    nb_iteration = int(np.ceil(nb_points/chunk_size))
    inv_transfo = np.linalg.inv(transfo)

    while nb_iteration > 0:
        max_position = min(current_position + chunk_size, nb_points)
        streamline = streamlines._data[current_position:max_position]

        # To access the deformation information, we need to go in voxel space
        streamline_vox = transform_streamlines(streamline,
                                               inv_transfo)

        current_streamline_vox = np.array(streamline_vox).T
        current_streamline_vox_list = current_streamline_vox.tolist()

        x_def = ndimage.map_coordinates(deformation_data[..., 0],
                                        current_streamline_vox_list, order=1)
        y_def = ndimage.map_coordinates(deformation_data[..., 1],
                                        current_streamline_vox_list, order=1)
        z_def = ndimage.map_coordinates(deformation_data[..., 2],
                                        current_streamline_vox_list, order=1)

        # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
        final_streamline = np.array([flip[0]*x_def, flip[1]*y_def, flip[2]*z_def])

        # The deformation obtained is in worldSpace
        if source == 'ants':
            final_streamline += np.array(streamline).T
        elif source == 'dipy':
            final_streamline += current_streamline_vox
            # The tractogram need to be brought back in world space to be saved
            final_streamline = transform_streamlines(final_streamline,
                                                     transfo)

        streamlines._data[current_position:max_position] \
            = final_streamline.T
        current_position = max_position
        nb_iteration -= 1


def get_streamlines_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max
