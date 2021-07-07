from collections import defaultdict
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import numpy as np
from connectivity_related_functions import endpoints_processing
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

def fa_extraction_use_cellinput(cell_streamlines,cell_id,fa_vol,N,voxel_size=None,affine=None):
    # input: groupsl, streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        fa_vol, fa values in the image domain
    #        N, # of ROIs
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    fa_group = defaultdict(list)
    
    idx = 0
    for i in range(1,N):
        for j in range(i+1,N):
            tmp_streamlines = cell_streamlines[idx]
            ROI_pair = cell_id[idx]
            idx = idx + 1;

            fa_streamlines = []
            for sl in tmp_streamlines:
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)

                #get FA for each streamlines
                ii, jj, kk = slpoints.T
                fa_value = fa_vol[ii, jj, kk]
                fa_streamlines.append(fa_value)
            fa_group[i,j].append(fa_streamlines)
#            if(len(fa_streamlines)>20):
#                pdb.set_trace()
    return fa_group

def fa_extraction_use_matinput(groupsl,fa_vol,N,voxel_size=None,affine=None,):
    # input: groupsl, streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        fa_vol, fa values in the image domain
    #        N, # of ROIs
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    fa_group = defaultdict(list)

    [N1,N2]=groupsl.shape
    
    for i in range(0,N1):
        for j in range(i+1,N2):
            tmp_streamlines = groupsl[i,j]

            Nfibers =0
            if(len(tmp_streamlines)>2):
                [tmp1,tmp2,Nfibers] = tmp_streamlines.shape


            fa_streamlines = []
            for n in range(0,Nfibers):
                tmp = tmp_streamlines[:,:,n]
                sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)

                #get FA for each streamlines
                ii, jj, kk = slpoints.T
                fa_value = fa_vol[ii, jj, kk]
                fa_streamlines.append(fa_value)
            fa_group[i,j].append(fa_streamlines)
    return fa_group

def fa_along_stmline_extraction(streamlines,fa_vol, voxel_size=None,affine=None,):
    # input: groupsl, streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        fa_vol, fa values in the image domain
    #        N, # of ROIs
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    fa_group = defaultdict(list)
    tmp_streamlines = list(streamlines)
    fa_streamlines = []
    for sl in tmp_streamlines:
        slpoints = _to_voxel_coordinates(sl, lin_T, offset)
        #get FA for each streamlines
        ii, jj, kk = slpoints.T
        fa_value = fa_vol[ii, jj, kk]
        fa_streamlines.append(fa_value)
    return  fa_streamlines


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
    endpointimg[i, j, k] = 1

    return endpointimg

def fa_mean_extraction(cm_fa_curve,N):
    # input: streamline matrix, number of nodes
    cm_fa_mean = np.zeros([N,N])
    cm_fa_max = np.zeros([N,N])
    cm_count = np.zeros([N,N])

    for i in range(0,N-1):
        for j in range(i+1,N):
            fa_curves = cm_fa_curve[i,j]
            
            if(len(fa_curves)>0):
                fa_curves = fa_curves[0]
            else:
                fa_curves = []

            if len(fa_curves)>0:
                temp_fa_curves = np.asarray(fa_curves)
                fa_mean_curve = np.mean(temp_fa_curves,axis = 0)
                cm_fa_mean[i,j] = np.mean(fa_mean_curve)
                cm_fa_max[i,j] = np.max(fa_mean_curve)
                cm_count[i,j] = len(fa_curves)
            else:
                cm_fa_max[i,j] = 0
                cm_fa_mean[i,j] = 0
                cm_count[i,j] = 0

    return  (cm_fa_mean,cm_fa_max,cm_count)







