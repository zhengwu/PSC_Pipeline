from collections import defaultdict
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import numpy as np
from numpy import ravel_multi_index
from scipy.stats import mode
from collections import Counter
from dipy.tracking.streamline import (length, set_number_of_points)
from numpy import linalg as LA
import copy
import nibabel as nib

import pdb


def fa_extraction(groupsl, fa_vol, N, voxel_size=None, affine=None, ):
    # input: groupsl, streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        fa_vol, fa values in the image domain
    #        N, # of ROIs
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    fa_group = defaultdict(list)

    for i in range(1, N):
        for j in range(i + 1, N):
            tmp_streamlines = groupsl[i, j]
            tmp_streamlines = list(tmp_streamlines)
            fa_streamlines = []
            for sl in tmp_streamlines:
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)

                # get FA for each streamlines
                ii, jj, kk = slpoints.T
                fa_value = fa_vol[ii, jj, kk]
                fa_streamlines.append(fa_value)
            fa_group[i, j].append(fa_streamlines)
    return fa_group


def nconnectivity_matrix_debug(streamlines, label_img, fiberlen_range, inROIlen, voxel_size=None,
                         affine=None, symmetric=False, return_mapping=False,
                         mapping_as_streamlines=False, keepfiberinroi=False):
    """return streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_img : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
    symmetric : bool, False by default (The current version do not output symmetric network anymore
                to save space)
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    return_mapping : bool, False by default
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.
    keepfiberinroi: bool, False by default
        If True, we keep fiber curves inside each ROI. Otherwise, we only keep fibers
        between two ROIs

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.

    """
    # Error checking on label_volume
    kind = label_img.dtype.kind
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_img.min() >= 0))
                       )
    valid_label_volume = (labels_positive and label_img.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)

    group = defaultdict(list)
    group_ma = defaultdict(list)
    mx = label_img.max() + 1
    matrix = np.zeros(shape=(mx, mx))

    # check the growth step of the streamlines, previouly only support step size 0.2 mm, now we support step size <1 mm
    tmpsl = streamlines[0]
    interval_dist = LA.norm(tmpsl[1:-1, :] - tmpsl[0:-2, :], axis=1)
    if ((interval_dist.min() < 0.01) | (interval_dist.max() > 0.4)):
        print "the streamline step size should be in the range (0.01,0.5) mm, reprocessing/resampling the streamlines..."

        # resampling the streamline
        arc_len = 0.2  # 0.2 m
        resampled_streamlines = []
        for sl in streamlines:
            flen = length(sl)
            Nstreamlines = int(flen / arc_len)
            line = set_number_of_points(sl, Nstreamlines)
            resampled_streamlines.append(line)

        streamlines = resampled_streamlines

    tmpsl = streamlines[0]
    interval_dist = LA.norm(tmpsl[1:-1, :] - tmpsl[0:-2, :], axis=1)
    if ((interval_dist.min() < 0.01) | (interval_dist.max() > 0.5)):
        print "seems resampling does not work, return empty connectivity matrix ....."
        return matrix, group

    npoints = inROIlen / ((interval_dist.min() + interval_dist.max()) / 2)  # number of points should be in the ROI
    print 'A connection will be established if both ends of a streamline have > ' + str(npoints) + ' points (' + str(
        inROIlen) + 'mm)' + ' inside ROIs. '
    if (npoints < 0.9): npoints = 1


    #for debugging
    #pdb.set_trace()

    #this is for debug
    roi_interested = [19]
    sl_roi2 = []
    slidx_roi2 = []
    streamline_idx = []
    streamline_unq = []
    idx = 0


    for sl in streamlines:
        slpoints = _to_voxel_coordinates(sl, lin_T, offset)
        ii, jj, kk = slpoints.T
        newlabel_img = label_img[ii,jj,kk]
        unq_label = np.unique(newlabel_img)

        streamline_idx.append(newlabel_img)
        streamline_unq.append(unq_label)

        if any(elem in unq_label for elem in roi_interested):
            sl_roi2.append(sl)
            slidx_roi2.append(newlabel_img)
            if keepfiberinroi:
                new_streamlines,num_sl,new_streamlines_startlabel,new_streamlines_endlabel = streamline_connectcut_returnfulllength(sl,newlabel_img,npoints,fiberlen_range)
            else:
                new_streamlines,num_sl,new_streamlines_startlabel,new_streamlines_endlabel = streamline_connectcut(sl,newlabel_img,npoints,fiberlen_range)

            if(num_sl==1):
                startroi = new_streamlines_startlabel[0]
                endroi = new_streamlines_endlabel[0]
                curr_streamline = np.squeeze(new_streamlines)

                # get the up triangular matrix and save the streamline
                if(startroi>endroi):
                    matrix[endroi,startroi] = matrix[endroi,startroi]+1
                    group[endroi,startroi].append(curr_streamline[::-1])
                else:
                    matrix[startroi,endroi] = matrix[startroi,endroi] + 1
                    group[startroi,endroi].append(curr_streamline)
            else:
                for i in range(0,num_sl):
                    startroi = new_streamlines_startlabel[i]
                    endroi = new_streamlines_endlabel[i]
                    curr_streamline = new_streamlines[i]

                    # get the up triangular matrix
                    if(startroi>endroi):
                        matrix[endroi,startroi] = matrix[endroi,startroi]+1
                        group[endroi,startroi].append(curr_streamline[::-1])
                    else:
                        matrix[startroi,endroi] = matrix[startroi,endroi] + 1
                        group[startroi,endroi].append(curr_streamline)

    print idx
    pdb.set_trace()

    return matrix, group, sl_roi2


    """"
    # for each streamline, we cut it and find how many pairs of rois it connects
    for sl in streamlines:
        slpoints = _to_voxel_coordinates(sl, lin_T, offset)
        ii, jj, kk = slpoints.T
        newlabel_img = label_img[ii, jj, kk]

        if keepfiberinroi:
            new_streamlines, num_sl, new_streamlines_startlabel, new_streamlines_endlabel = streamline_connectcut_returnfulllength(
                sl, newlabel_img, npoints, fiberlen_range)
        else:
            new_streamlines, num_sl, new_streamlines_startlabel, new_streamlines_endlabel = streamline_connectcut(sl,
                                                                                                                  newlabel_img,
                                                                                                                  npoints,
                                                                                                                  fiberlen_range)
        if (num_sl == 1):
            startroi = new_streamlines_startlabel[0]
            endroi = new_streamlines_endlabel[0]
            curr_streamline = np.squeeze(new_streamlines)

            # get the up triangular matrix and save the streamline
            if (startroi > endroi):
                matrix[endroi, startroi] = matrix[endroi, startroi] + 1
                group[endroi, startroi].append(curr_streamline[::-1])
            else:
                matrix[startroi, endroi] = matrix[startroi, endroi] + 1
                group[startroi, endroi].append(curr_streamline)
        else:
            for i in range(0, num_sl):
                startroi = new_streamlines_startlabel[i]
                endroi = new_streamlines_endlabel[i]
                curr_streamline = new_streamlines[i]

                # get the up triangular matrix
                if (startroi > endroi):
                    matrix[endroi, startroi] = matrix[endroi, startroi] + 1
                    group[endroi, startroi].append(curr_streamline[::-1])
                else:
                    matrix[startroi, endroi] = matrix[startroi, endroi] + 1
                    group[startroi, endroi].append(curr_streamline)

    if return_mapping:
        # Return the mapping matrix and the mapping
        return matrix, group
    else:
        return matrix
    """


def nconnectivity_matrix_selected(streamlines, list_selected_slind, label_img, fiberlen_range, inROIlen,
                                  voxel_size=None,
                                  affine=None, symmetric=True, return_mapping=False,
                                  mapping_as_streamlines=False):
    """return streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_img : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
    symmetric : bool, False by default
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    return_mapping : bool, False by default
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.

    """
    # Error checking on label_volume
    kind = label_img.dtype.kind
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_img.min() >= 0))
                       )
    valid_label_volume = (labels_positive and label_img.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)

    mx = label_img.max() + 1
    matrix = np.zeros(shape=(mx, mx))
    # check the growth step of the streamlines, previouly only support step size 0.2 mm, now we support step size <1 mm
    tmpsl = streamlines[0]
    interval_dist = LA.norm(tmpsl[1:-1, :] - tmpsl[0:-2, :], axis=1)
    if ((interval_dist.min() < 0.01) | (interval_dist.max() > 1)):
        print " the step size should be in the range(0.01,1)mm, reprocessing/resampling the streamlines..."

    # resampling the streamline
    arc_len = 0.2  # 0.2 m
    resampled_streamlines = []
    for sl in streamlines:
        flen = length(sl)
        Nstreamlines = int(flen / arc_len)
        line = set_number_of_points(sl, Nstreamlines)
        resampled_streamlines.append(line)

    streamlines = resampled_streamlines

    tmpsl = streamlines[0]
    interval_dist = LA.norm(tmpsl[1:-1, :] - tmpsl[0:-2, :], axis=1)
    if ((interval_dist.min() < 0.01) | (interval_dist.max() > 1)):
        print "resampling does not work, return empty connectivity matrix ....."
        return matrix, group

    npoints = inROIlen / ((interval_dist.min() + interval_dist.max()) / 2)  # number of points should be in the ROI
    print 'A connection will be established if both ends of a streamline have > ' + str(npoints) + ' points (' + str(
        inROIlen) + 'mm)' + ' inside ROIs. '
    if (npoints < 0.9): npoints = 1

    # for each streamline, we cut it and find how many pairs of rois it connects
    for selected_idx in list_selected_slind:
        sl = streamlines[selected_idx]
        slpoints = _to_voxel_coordinates(sl, lin_T, offset)
        ii, jj, kk = slpoints.T
        newlabel_img = label_img[ii, jj, kk]
        new_streamlines, num_sl, new_streamlines_startlabel, new_streamlines_endlabel = streamline_connectcut(sl,
                                                                                                              newlabel_img,
                                                                                                              npoints,
                                                                                                              fiberlen_range)
        if (num_sl == 1):
            startroi = new_streamlines_startlabel[0]
            endroi = new_streamlines_endlabel[0]
            curr_streamline = np.squeeze(new_streamlines)

            # get the up triangular matrix
            if (startroi > endroi):
                matrix[endroi, startroi] = matrix[endroi, startroi] + 1
            else:
                matrix[startroi, endroi] = matrix[startroi, endroi] + 1
        else:
            for i in range(0, num_sl):
                startroi = new_streamlines_startlabel[i]
                endroi = new_streamlines_endlabel[i]
                curr_streamline = new_streamlines[i]

                # get the up triangular matrix
                if (startroi > endroi):
                    matrix[endroi, startroi] = matrix[endroi, startroi] + 1
                else:
                    matrix[startroi, endroi] = matrix[startroi, endroi] + 1

    if return_mapping:
        # Return the mapping matrix and the mapping
        return matrix
    else:
        return matrix


def streamline_pruning(streamlines, label_volume, voxel_size=None,
                       affine=None):
    """Cut streamlines such that it only connects two regions of interests

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
    """

    # Error checking on label_volume
    kind = label_volume.dtype.kind
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_volume.min() >= 0))
                       )
    valid_label_volume = (labels_positive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    streamlines = list(streamlines)
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    # for each streamlines, we will check it and cut it.
    new_streamlines = []
    for sl in streamlines:
        # Map the streamlines coordinates to voxel coordinates
        sl_volidx = _to_voxel_coordinates(sl, lin_T, offset)
        sl_labels = label_volume[sl_volidx[:, 0], sl_volidx[:, 1], sl_volidx[:, 2]]
        temp_streamlines, num_sl = streamline_cut(sl, sl_labels)
        if (num_sl == 1):
            curr_sl = temp_streamlines
            new_streamlines.append(curr_sl)
        else:
            for i in range(0, num_sl):
                curr_sl = temp_streamlines[i]
                new_streamlines.append(curr_sl)

    return new_streamlines


def streamline_cut(streamline, streamline_labels):
    unq_label = np.unique(streamline_labels)
    Npoint = len(streamline_labels)
    num_sl = 0
    new_streamlines = []

    # case1: all streamline is in the wm or gm
    if (len(unq_label) == 1 | unq_label[0] > 0):
        num_sl = num_sl + 1
        return streamline, num_sl

    zero_label = np.squeeze(np.where(streamline_labels == 0))
    if (len(np.atleast_1d(zero_label)) > 5):  # if more than 5 points are in wm, we segment it
        grad_zero_label = np.gradient(zero_label)
    else:  # if not, we don't process it
        num_sl = num_sl + 1
        return streamline, num_sl

    segind = np.where(grad_zero_label > 1)
    newsegidx = np.append(0, segind)
    newsegidx = np.append(newsegidx, len(zero_label) - 1)

    nseg = len(newsegidx) / 2
    new_streamlines = []
    for i in range(0, nseg):
        startidx = zero_label[newsegidx[2 * i]]
        endidx = zero_label[newsegidx[2 * i + 1]]
        for j in range(5, 1, -1):
            aa = max(0, startidx - j)
            bb = min(Npoint - 1, endidx + j) - 1
            if ((len(np.unique(streamline_labels[aa:startidx])) == 1) & (
                    len(np.unique(streamline_labels[endidx + 1:bb + 1])) == 1)):
                tempseglabel = range(aa, bb + 1)
                if (len(tempseglabel) > 8):
                    new_streamlines.append(streamline[tempseglabel])
                    num_sl = num_sl + 1
                    break

    new_streamlines = [t for t in new_streamlines]
    f_streamlines = []
    for sl in new_streamlines:
        sl = np.squeeze(np.asarray(sl))
        # get fibers having length between 10mm and 200mm
        f_streamlines.append(sl)

    return f_streamlines, num_sl


def ndbincount(x, weights=None, shape=None):
    """Like bincount, but for nd-indicies.

    Parameters
    ----------
    x : array_like (N, M)
        M indices to a an Nd-array
    weights : array_like (M,), optional
        Weights associated with indices
    shape : optional
        the shape of the output
    """
    x = np.asarray(x)
    if shape is None:
        shape = x.max(1) + 1

    x = ravel_multi_index(x, shape)
    # out = np.bincount(x, weights, minlength=np.prod(shape))
    # out.shape = shape
    # Use resize to be compatible with numpy < 1.6, minlength new in 1.6

    out = np.bincount(x, weights)
    out.resize(shape)

    return out


def endpoints_processing(endpoint, label_volume):
    # process the endpoint
    i, j, k = endpoint.T
    startpoint = np.array([i[0], j[0], k[0]])
    endpoint = np.array([i[1], j[1], k[1]])

    startplabel = label_volume[i[0], j[0], k[0]]
    endplabel = label_volume[i[1], j[1], k[1]]

    if startplabel == 0:
        startplabel = findendingroi(startpoint, label_volume, 2)

    if endplabel == 0:
        endplabel = findendingroi(endpoint, label_volume, 2)

    endlabel = [startplabel, endplabel]
    return endlabel


def findendingroi(spoint, label_volume, wind_thrd):
    ''''
    find the ending roi for fibers end or start with 0
    spoint - starting or ending point
    label_volume - labeled volume
    wind_thrd - search region radio
    '''
    # test
    idx = spoint

    alls_i = np.array([], dtype='int')
    alls_j = np.array([], dtype='int')
    alls_k = np.array([], dtype='int')
    N = 2 * wind_thrd + 1
    hN = wind_thrd
    vol_dim = label_volume.shape

    for i in range(0, N):
        if (np.all(np.logical_and(idx - hN + i < vol_dim, idx - hN + i > -0.1))):
            alls_i = np.append(alls_i, idx[0] - hN + i)
            alls_j = np.append(alls_j, idx[1] - hN + i)
            alls_k = np.append(alls_k, idx[2] - hN + i)

    labels = label_volume[alls_i, :, :][:, alls_j, :][:, :, alls_k]

    # get the freq of each label
    nonzero_idxi, nonzero_idxj, nonzero_idxk = np.nonzero(np.asarray(labels))
    flat_non_zero = np.asarray(labels[nonzero_idxi, nonzero_idxj, nonzero_idxk])

    if flat_non_zero.size == 0:
        return 0
    else:
        index = mode(flat_non_zero, axis=None)
        return index[0][0]


def streamline_zerolab_endpoints(streamlines, label_volume, voxel_size=None, affine=None):
    # input: streamlines, input streamlines
    #        anaimg, input image
    endpointimg = np.zeros(label_volume.shape, dtype='int')

    # get the ending points of streamlines
    streamlines = list(streamlines)
    endpoints = [sl[0::len(sl) - 1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    # process the endpoints
    Nstreamlines = len(endpoints)
    endlabels = np.zeros((2, Nstreamlines), dtype=int)
    for i in range(0, Nstreamlines):
        endpoint = endpoints[i]
        endlabel = endpoints_processing(endpoint, label_volume)
        if (endlabel[0] == 0):
            endpointimg[endpoint[0][0], endpoint[0][1], endpoint[0][2]] = 1

        if (endlabel[1] == 0):
            endpointimg[endpoint[1][0], endpoint[1][1], endpoint[1][2]] = 1

    return endpointimg


def streamline_endpoints(streamlines, label_volume, voxel_size=None, affine=None):
    # input: streamlines, input streamlines
    #        anaimg, input image
    endpointimg = np.zeros(label_volume.shape, dtype='int')

    # get the ending points of streamlines
    streamlines = list(streamlines)
    endpoints = [sl[0::len(sl) - 1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    i, j, k = endpoints.T

    # processing the i,j,k, because of the offset, some endpoints are mapped
    # to the outside of the brain
    dim = label_volume.shape
    if (np.max(i) > dim[0] and np.max(j) > dim[1] and np.max(k) > dim[2]):
        raise IndexError('streamline has points that map to outside of the brain voxel')
    i[np.where(i > dim[0] - 1)] = dim[0] - 1
    j[np.where(j > dim[1] - 1)] = dim[1] - 1
    k[np.where(k > dim[2] - 1)] = dim[2] - 1

    # pdb.set_trace()
    endpointimg[i, j, k] = 1

    return endpointimg


def cortexband_dilation_endpoints(filtered_labels, streamline_endpoints):
    # input: filtered_labels, original brain pacellation, usually the output from
    # freesurf after sorting the index;
    # streamline_endpoints: endpoints of the streamlines
    # return new_filtered_labels

    new_filtered_labels = filtered_labels

    # all points have value=1 in the streamline_endpoints image
    dim_i, dim_j, dim_k = np.nonzero(streamline_endpoints)
    Nvol = len(dim_i)

    # for each voxel in streamline_endpoints
    for i in range(0, Nvol):
        idx = np.array([dim_i[i], dim_j[i], dim_k[i]])
        label = cluster_endpoints(idx, filtered_labels, 3, 4)
        new_filtered_labels[idx[0], idx[1], idx[2]] = label
    return new_filtered_labels


def cortexband_dilation_wm(filtered_labels, full_labels, dilation_para):
    """ dilate the cortical band using wm region

    # input: filtered_labels, original brain pacellation, usually the output from
    # freesurf after sorting the index;
    # full_labels: full_labels of the parcellation
    # return new_filtered_labels
    """
    new_filtered_labels = copy.deepcopy(filtered_labels)

    wm_labels = np.zeros(full_labels.shape, dtype='int')
    wm_labels[full_labels == 2] = 1  # Left-Cerebral-White-Matter
    wm_labels[full_labels == 41] = 1  # Right-Cerebral-White-Matter
    wm_labels[(full_labels > 2999) & (full_labels < 5005)] = 1

    # all points have value=1 in the streamline_endpoints image
    dim_i, dim_j, dim_k = np.nonzero(wm_labels)
    Nvol = len(dim_i)

    # for each voxel in streamline_endpoints
    dsize = dilation_para[0]
    wsize = dilation_para[1]

    for i in range(0, Nvol):
        idx = np.array([dim_i[i], dim_j[i], dim_k[i]])
        label = cluster_endpoints(idx, filtered_labels, dsize, wsize)
        new_filtered_labels[idx[0], idx[1], idx[2]] = label

    return new_filtered_labels


def cluster_endpoints(idx, filtered_labels, dist_thrd, wind_thrd):
    # input: idx, index of the current voxel
    # filtered_labels: original brain pacellation, usually the output from
    # freesurf after sorting the index;
    # dist_thrd: distance threshold
    # wind_thrd: window threshold
    # return newlabels

    label = filtered_labels[idx[0], idx[1], idx[2]]

    # if current idx already has label in original pacellation, return the original label
    if label > 0:
        return label

    # else, we need to clustering current voxel
    vol_dim = filtered_labels.shape

    # build a window with size (2*wind_thrd+1, 2*wind_thrd+1, 2*wind_thrd+1)
    alls_i = np.array([], dtype='int')
    alls_j = np.array([], dtype='int')
    alls_k = np.array([], dtype='int')

    N = 2 * wind_thrd + 1
    hN = wind_thrd
    for i in range(0, N):
        if (np.all(np.logical_and(idx - hN + i < vol_dim, idx - hN + i > -0.1))):
            alls_i = np.append(alls_i, idx[0] - hN + i)
            alls_j = np.append(alls_j, idx[1] - hN + i)
            alls_k = np.append(alls_k, idx[2] - hN + i)

    labels_ori = filtered_labels[alls_i, :, :][:, alls_j, :][:, :, alls_k]

    nonzero_idxi, nonzero_idxj, nonzero_idxk = np.nonzero(np.asarray(labels_ori))
    flat_nonzero_value = np.asarray(labels_ori[nonzero_idxi, nonzero_idxj, nonzero_idxk])

    # get the freq of each label
    unique, counts = np.unique(flat_nonzero_value, return_counts=True)
    dtype = [('label', int), ('freq', int)]
    values = [(unique[i], counts[i]) for i in range(0, len(unique))]
    val_hist = np.array(values, dtype=dtype)
    sorted_val = np.sort(val_hist, order='freq')
    ascend_sort_val = sorted_val[::-1]

    for pair in ascend_sort_val:
        label_tmp = pair[0]
        coord_i, coord_j, coord_k = np.where(labels_ori == label_tmp)
        coord_i = coord_i - hN
        coord_j = coord_j - hN
        coord_k = coord_k - hN
        dist_c = np.sqrt(np.square(coord_i) + np.square(coord_j) + np.square(coord_k))
        if ((np.min(dist_c) < (dist_thrd + 0.1)) & ((pair[1].astype('float') / (N * N * N)) > 0.15)):
            label = label_tmp
            return label

    # return label = 0
    return label


def streamline_connectcut(streamline, streamline_labels, npoints, fiberlen_range):
    """ cut streamline into streamlines that connecting different rois

    # input: streamline: one streamline
    #        streamline_label: the labels along streamline
    #        npoints: threthhold
    """
    unq_label = np.unique(streamline_labels)
    Nrois = len(streamline_labels)
    num_sl = 0
    new_streamlines = []
    new_streamlines_startlabel = []
    new_streamlines_endlabel = []

    # case1: the streamline is in the wm or only connect two rois, we just return this streamline
    if (len(unq_label) == 1):
        num_sl = num_sl + 1
        new_streamlines_startlabel.append(unq_label[0])
        new_streamlines_endlabel.append(unq_label[0])
        return streamline, num_sl, new_streamlines_startlabel, new_streamlines_endlabel

    if (len(unq_label) == 2):
        new_streamlines_startlabel.append(streamline_labels[0])
        new_streamlines_endlabel.append(streamline_labels[-1])
        num_sl = num_sl + 1
        return streamline, num_sl, new_streamlines_startlabel, new_streamlines_endlabel

    # case2: the streamline connects multiple rois
    ct = Counter(streamline_labels)
    passed_roi = []
    for t in ct:
        if ((t != 0) & (ct[t] > npoints)):
            passed_roi.append(t)

    # cut the streamline into nchoose(len(passed_roi),2) pieces
    for i in range(0, len(passed_roi)):
        for j in range(i + 1, len(passed_roi)):
            roia = passed_roi[i]
            roib = passed_roi[j]
            # find the part connects roia and roib
            label_roia = np.squeeze(np.asarray(np.where(streamline_labels == roia)))
            label_roib = np.squeeze(np.asarray(np.where(streamline_labels == roib)))
            if (label_roia[0] < label_roib[0]):  # if roia is in front of roib
                startidx = label_roia[-1]  # start index
                endidx = label_roib[1]  #
                tmpsl = streamline[startidx:endidx]
                if (length(tmpsl) > fiberlen_range[0]):  # for streamlines longer than xx mm, we record it
                    new_streamlines.append(tmpsl)
                    new_streamlines_startlabel.append(roia)
                    new_streamlines_endlabel.append(roib)
                    num_sl = num_sl + 1
            else:
                startidx = label_roib[-1]
                endidx = label_roia[1]  # can be improved here
                tmpsl = streamline[startidx:endidx]
                if (length(tmpsl) > fiberlen_range[0]):  # for streamlines longer than xx mm, we keep it
                    new_streamlines.append(tmpsl)
                    new_streamlines_startlabel.append(roib)
                    new_streamlines_endlabel.append(roia)
                    num_sl = num_sl + 1

    return new_streamlines, num_sl, new_streamlines_startlabel, new_streamlines_endlabel


def streamline_connectcut_returnfulllength(streamline, streamline_labels, npoints, fiberlen_range):
    """ cut streamline into streamlines that connecting different rois, here we try to return
        the full length of the fiber. Note in the function of streamline_connectcut, we only
        return the intermedia part of a fiber between two rois. Here we retrun strealimes within
        the ROIs.

    # input: streamline: one 3D streamline
    #        streamline_label: the labels along streamline
    #        npoints: number of points in ROI
    """
    unq_label = np.unique(streamline_labels)
    Nrois = len(streamline_labels)
    num_sl = 0
    new_streamlines = []
    new_streamlines_startlabel = []
    new_streamlines_endlabel = []

    # case1: the streamline is in the wm or only connect two rois, we just return this streamline
    if (len(unq_label) == 1):
        num_sl = num_sl + 1
        new_streamlines_startlabel.append(unq_label[0])
        new_streamlines_endlabel.append(unq_label[0])
        return streamline, num_sl, new_streamlines_startlabel, new_streamlines_endlabel

    if (len(unq_label) == 2):
        new_streamlines_startlabel.append(streamline_labels[0])
        new_streamlines_endlabel.append(streamline_labels[-1])
        num_sl = num_sl + 1
        return streamline, num_sl, new_streamlines_startlabel, new_streamlines_endlabel

    # case2: the streamline connects multiple rois
    ct = Counter(streamline_labels)
    passed_roi = []
    for t in ct:
        if ((t != 0) & (ct[t] > npoints)):  # remove 0
            passed_roi.append(t)

    # cut the streamline into nchoose(len(passed_roi),2) pieces
    for i in range(0, len(passed_roi)):
        for j in range(i + 1, len(passed_roi)):
            roia = passed_roi[i]
            roib = passed_roi[j]
            # find the part connects roia and roib
            label_roia = np.squeeze(np.asarray(np.where(streamline_labels == roia)))
            label_roib = np.squeeze(np.asarray(np.where(streamline_labels == roib)))
            if (label_roia[0] < label_roib[0]):  # if roia is in front of roib
                startidx = label_roia[0]  # start index
                endidx = label_roib[-1]  #
                tmpsl = streamline[startidx:endidx]
                if (length(tmpsl) > fiberlen_range[0]):  # for streamlines longer than xx mm, we record it
                    new_streamlines.append(tmpsl)
                    new_streamlines_startlabel.append(roia)
                    new_streamlines_endlabel.append(roib)
                    num_sl = num_sl + 1
            else:
                startidx = label_roib[1]
                endidx = label_roia[-1]  # can be improved here
                tmpsl = streamline[startidx:endidx]
                if (length(tmpsl) > fiberlen_range[0]):  # for streamlines longer than xx mm, we keep it
                    new_streamlines.append(tmpsl)
                    new_streamlines_startlabel.append(roib)
                    new_streamlines_endlabel.append(roia)
                    num_sl = num_sl + 1

    return new_streamlines, num_sl, new_streamlines_startlabel, new_streamlines_endlabel