from collections import defaultdict
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import numpy as np
from numpy import (asarray, ceil, dot, empty, eye, sqrt)
from numpy import ravel_multi_index
from scipy.stats import mode
import pdb

def fa_extraction(groupsl,fa_vol,N, voxel_size=None,affine=None,):
    # input: groupsl, streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        fa_vol, fa values in the image domain
    #        N, # of ROIs
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    fa_group = defaultdict(list)

    for i in range(1,N):
        for j in range(i+1,N):
            tmp_streamlines = groupsl[i,j]
            tmp_streamlines = list(tmp_streamlines)
            fa_streamlines = []
            for sl in tmp_streamlines:
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)

                #get FA for each streamlines
                ii, jj, kk = slpoints.T
                fa_value = fa_vol[ii, jj, kk]
                fa_streamlines.append(fa_value)
            fa_group[i,j].append(fa_streamlines)
    return fa_group

def nconnectivity_matrix(streamlines, label_volume, voxel_size=None,
                        affine=None, symmetric=True, return_mapping=False,
                        mapping_as_streamlines=False):
    """Counts the streamlines that start and end at each label pair.

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
    kind = label_volume.dtype.kind
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_volume.min() >= 0))
                      )
    valid_label_volume = (labels_positive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)

    #take the first and last point of each streamline
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]
    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    #process the endpoints
    Nstreamlines = len(endpoints)
    endlabels = np.zeros((2,Nstreamlines),dtype=int)
    for i in range(0,Nstreamlines):
        endpoint = endpoints[i]
        endlabel = endpoints_processing(endpoint,label_volume)
        endlabels[:,i] = endlabel

    #get labels for label_volume
    #i, j, k = endpoints.T
    #endlabels = label_volume[i, j, k]


    #if symmetric:
    #    endlabels.sort(0)
    mx = label_volume.max() + 1

    matrix = ndbincount(endlabels, shape=(mx, mx))
    if symmetric:
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix,np.diagonal(matrix)/2)

    if return_mapping:
        mapping = defaultdict(list)
        for i, (a, b) in enumerate(endlabels.T):
            mapping[a, b].append(i)

        # Replace each list of indices with the streamlines they index
        group = defaultdict(list)
        if mapping_as_streamlines:
            for key in mapping:
                group[key] = [streamlines[i] for i in mapping[key]]
            for key in mapping:
                if key[0]>key[1]:
                    #merge low triangle matrix in group ( when key[0] > key[1])
                    tmp = group[key[0],key[1]]
                    low_tri_streamlines = []
                    for sl in tmp:
                        low_tri_streamlines.append(sl[::-1])
                    group[key[1],key[0]] = group[key[1],key[0]] + low_tri_streamlines
                    group[key[0],key[1]] = []

        # Return the mapping matrix and the mapping
        return matrix, group
    else:
        return matrix

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

def endpoints_processing(endpoint,label_volume):
    #process the endpoint
    i,j,k = endpoint.T
    startpoint = np.array([i[0],j[0],k[0]])
    endpoint = np.array([i[1],j[1],k[1]])

    startplabel = label_volume[i[0],j[0],k[0]]
    endplabel = label_volume[i[1],j[1],k[1]]

    if startplabel == 0:
        startplabel = findendingroi(startpoint,label_volume,2)

    if endplabel == 0:
        endplabel = findendingroi(endpoint,label_volume,2)

    endlabel = [startplabel,endplabel]
    return endlabel


def findendingroi(spoint,label_volume, wind_thrd):
    #test
    idx = spoint

    alls_i = np.array([],dtype='int')
    alls_j = np.array([],dtype='int')
    alls_k = np.array([],dtype='int')
    N = 2*wind_thrd+1
    hN = wind_thrd
    vol_dim = label_volume.shape

    for i in range(0,N):
        if ( np.all(np.logical_and(idx -hN + i < vol_dim,idx-hN+i>-0.1)) ):
            alls_i = np.append(alls_i,idx[0]-hN+i)
            alls_j = np.append(alls_j,idx[1]-hN+i)
            alls_k = np.append(alls_k,idx[2]-hN+i)

    labels = label_volume[alls_i,:,:][:,alls_j,:][:,:,alls_k]

    #get the freq of each label
    nonzero_idxi,nonzero_idxj,nonzero_idxk = np.nonzero(np.asarray(labels))
    flat_non_zero = np.asarray(labels[nonzero_idxi,nonzero_idxj,nonzero_idxk])

    if flat_non_zero.size==0:
        return 0
    else:
        index = mode(flat_non_zero,axis=None)
        return index[0][0]

def streamline_zerolab_endpoints(streamlines,label_volume,voxel_size=None,affine=None):
    # input: streamlines, input streamlines
    #        anaimg, input image
    endpointimg = np.zeros(label_volume.shape, dtype='int')

    # get the ending points of streamlines
    streamlines = list(streamlines)
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

     #process the endpoints
    Nstreamlines = len(endpoints)
    endlabels = np.zeros((2,Nstreamlines),dtype=int)
    for i in range(0,Nstreamlines):
        endpoint = endpoints[i]
        endlabel = endpoints_processing(endpoint,label_volume)
        if(endlabel[0] == 0):
            endpointimg[endpoint[0][0],endpoint[0][1],endpoint[0][2]]=1

        if(endlabel[1] == 0):
            endpointimg[endpoint[1][0],endpoint[1][1],endpoint[1][2]]=1

    return endpointimg

def streamline_endpoints(streamlines,label_volume,voxel_size=None,affine=None):
    # input: streamlines, input streamlines
    #        anaimg, input image
    endpointimg = np.zeros(label_volume.shape, dtype='int')

    # get the ending points of streamlines
    streamlines = list(streamlines)
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    
    i, j, k = endpoints.T
    
    #processing the i,j,k, because of the offset, some endpoints are mapped
    #to the outside of the brain 
    dim = label_volume.shape
    if(np.max(i)>dim[0] and np.max(j)>dim[1] and np.max(k)>dim[2]):
        raise IndexError('streamline has points that map to outside of the brain voxel')
    i[np.where(i>dim[0]-1)] = dim[0]-1
    j[np.where(j>dim[1]-1)] = dim[1]-1
    k[np.where(k>dim[2]-1)] = dim[2]-1

    #pdb.set_trace()
    endpointimg[i, j, k] = 1

    return endpointimg

def cortexband_dilation(filtered_labels,streamline_endpoints):
    # input: filtered_labels, original brain pacellation, usually the output from
    # freesurf after sorting the index;
    # streamline_endpoints: endpoints of the streamlines
    # return new_filtered_labels

    new_filtered_labels = filtered_labels

    # all points have value=1 in the streamline_endpoints image
    dim_i,dim_j,dim_k = np.nonzero(streamline_endpoints)
    Nvol = len(dim_i)

    # for each voxel in streamline_endpoints
    for i in range(0,Nvol):
        idx = np.array([dim_i[i],dim_j[i],dim_k[i]])
        label = cluster_endpoints(idx,filtered_labels,3,4)
        new_filtered_labels[idx[0],idx[1],idx[2]] = label
    return new_filtered_labels

def cluster_endpoints(idx,filtered_labels,dist_thrd,wind_thrd):
    # input: idx, index of the current voxel
    # filtered_labels: original brain pacellation, usually the output from
    # freesurf after sorting the index;
    # dist_thrd: distance threshold 
    # wind_thrd: window threshold
    # return newlabels

    label = filtered_labels[idx[0],idx[1],idx[2]]
     
    # if current idx already has label in original pacellation, return the original label
    if label >0:
        return label
    
    # else, we need to clustering current voxel
    vol_dim = filtered_labels.shape

    # build a window with size (2*wind_thrd+1, 2*wind_thrd+1, 2*wind_thrd+1)   
    alls_i = np.array([],dtype='int')
    alls_j = np.array([],dtype='int')
    alls_k = np.array([],dtype='int')

    N = 2*wind_thrd+1
    hN = wind_thrd
    for i in range(0,N):
        if ( np.all(np.logical_and(idx -hN + i < vol_dim,idx-hN+i>-0.1)) ):
            alls_i = np.append(alls_i,idx[0]-hN+i)
            alls_j = np.append(alls_j,idx[1]-hN+i)
            alls_k = np.append(alls_k,idx[2]-hN+i)

    labels_ori = filtered_labels[alls_i,:,:][:,alls_j,:][:,:,alls_k]

    nonzero_idxi,nonzero_idxj,nonzero_idxk = np.nonzero(np.asarray(labels_ori))
    flat_nonzero_value = np.asarray(labels_ori[nonzero_idxi,nonzero_idxj,nonzero_idxk])

    #get the freq of each label
    unique, counts = np.unique(flat_nonzero_value, return_counts=True)
    dtype = [('label',int),('freq',int)]
    values = [(unique[i],counts[i]) for i in range(0,len(unique))]
    val_hist = np.array(values,dtype=dtype)
    sorted_val = np.sort(val_hist,order='freq')
    ascend_sort_val = sorted_val[::-1]

    for pair in ascend_sort_val:
        label_tmp = pair[0]
        coord_i,coord_j,coord_k = np.where(labels_ori==label_tmp)
        coord_i = coord_i - hN
        coord_j = coord_j - hN
        coord_k = coord_k - hN
        dist_c = np.sqrt(np.square(coord_i)+np.square(coord_j)+np.square(coord_k))
        if (np.min(dist_c) < (dist_thrd+0.1)):
            label = label_tmp
            return label

    # have to return label = 0
    return label
                

