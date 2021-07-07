from collections import defaultdict
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import numpy as np
from connectivity_related_functions import endpoints_processing
from dipy.tracking.streamline import length
import pdb

def roi_volume_extraction(label_img,N):
    # input: label_img, image of labels, index from 0 - N
    #        N, # of ROIs
    ROI_vol = np.zeros(N)
    for i in range(1,N+1):   # starting from 1 because 0 is WM
        xx,yy,zz = np.where(label_img==i)
        ROI_vol[i-1] = len(xx)
    return  ROI_vol

def roi_volume_extraction_cellinput(label_img,N):
    # input: label_img, image of labels, index from 0 - N
    #        N, # of ROIs
    ROI_vol = np.zeros(N)
    for i in range(1,N):   # starting from 1 because 0 is WM
        xx,yy,zz = np.where(label_img==i)
        ROI_vol[i] = len(xx)
    return  ROI_vol

def rois_connectedvol_matinput(label_img,N,groupsl,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    ROI_vol = roi_volume_extraction(label_img,N)
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    connectedvolratio_cm = np.zeros([N,N])
    connectedvol_cm = np.zeros([N,N])
    [N1,N2]=groupsl.shape

    for i in range(0,N1-1):              #starting from 0
        for j in range(i+1,N2):
            tmp_streamlines = groupsl[i,j]

            Nfibers =0
            if(len(tmp_streamlines)>2):
                [tmp1,tmp2,Nfibers] = tmp_streamlines.shape

            tmpstartp = []
            tmpendp = []
            for n in range(0,Nfibers):
                tmp = tmp_streamlines[:,:,n]
                sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)
                ii, jj, kk = slpoints.T
                newlabel_img = label_img[ii,jj,kk]
                tmpstartp.append(ii[1]*1000000+jj[1]*1000+kk[1])
                tmpendp.append(ii[-1]*1000000+jj[-1]*1000+kk[-1])
            nstartp_ratio = len(list(set(tmpstartp)))/ (ROI_vol[i])
            nendp_ratio = len(list(set(tmpendp)))/(ROI_vol[j])
            nstartp = len(list(set(tmpstartp)))
            nendp = len(list(set(tmpendp)))
            connectedvol_cm[i,j] = (nstartp + nendp)
            connectedvolratio_cm[i,j] = (nstartp_ratio + nendp_ratio)/2

    return connectedvol_cm,connectedvolratio_cm

def rois_connectedvol_cellinput(label_img,N,cell_streamlines,cell_id,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    ROI_vol = roi_volume_extraction_cellinput(label_img,N)
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    connectedvolratio_cm = np.zeros([N,N])
    connectedvol_cm = np.zeros([N,N])
    
    idx = 0
    for i in range(1,N):
        for j in range(i+1,N):
            tmp_streamlines = cell_streamlines[idx]
            idx = idx + 1
            
            tmpstartp = []
            tmpendp = []
            for sl in tmp_streamlines:
                #sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)
                ii, jj, kk = slpoints.T
                newlabel_img = label_img[ii,jj,kk]
                tmpstartp.append(ii[1]*1000000+jj[1]*1000+kk[1])
                tmpendp.append(ii[-1]*1000000+jj[-1]*1000+kk[-1])
            nstartp_ratio = len(list(set(tmpstartp)))/ (ROI_vol[i])
            nendp_ratio = len(list(set(tmpendp)))/(ROI_vol[j])
            nstartp = len(list(set(tmpstartp)))
            nendp = len(list(set(tmpendp)))
            connectedvol_cm[i,j] = (nstartp + nendp)
            connectedvolratio_cm[i,j] = (nstartp_ratio + nendp_ratio)/2

    return connectedvol_cm,connectedvolratio_cm

def rois_fiberlen_cellinput(N,cell_streamlines):
    # input:
    #        N: # of ROIs
    #        cell_streamlines: streamlines connecting ith and jth ROIs
    #        voxel_size, affine: some parameters to transfore streamlines to voxel

    connectcm_len = np.zeros([N,N])
    
    idx = 0
    for i in range(1,N):
        for j in range(i+1,N):
            tmp_streamlines = cell_streamlines[idx]
            idx = idx + 1
            
            tmp_len = 0
            for sl in tmp_streamlines:
                flen = length(sl)
                tmp_len = tmp_len + flen
            
            if(len(tmp_streamlines)>0):
                connectcm_len[i,j] = tmp_len/len(tmp_streamlines)


    return connectcm_len


def tworoi_inters_streamlines_matinput(label_img,N,groupsl,roia,roib,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    roia_roib_stls = []
    for i in range(0,N-1):
        for j in range(i+1,N):
            tmp_streamlines = groupsl[i,j]
            [tmp1,tmp2,Nfibers] = tmp_streamlines.shape
            for n in range(0,Nfibers):
                tmp = tmp_streamlines[:,:,n]
                sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)
                ii, jj, kk = slpoints.T

                newlabel_img = label_img[ii,jj,kk]
                if ((roia in newlabel_img) & (roib in newlabel_img)):
                    roia_roib_stls.append(sl)
                    print newlabel_img

    return roia_roib_stls


def rois_countmatrix_matinput(label_img,N,groupsl,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    countm = np.zeros([N+1,N+1])

    for i in range(0,N-1):
        for j in range(i+1,N):
            tmp_streamlines = groupsl[i,j]
            [tmp1,tmp2,Nfibers] = tmp_streamlines.shape
            for n in range(0,Nfibers):
                tmp = tmp_streamlines[:,:,n]
                sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)
                ii, jj, kk = slpoints.T

                newlabel_img = label_img[ii,jj,kk]
                roi_id = list(set(newlabel_img))
                Nid = len(roi_id)
                for aa in range(1,Nid-1):
                    for bb in range(aa+1,Nid):
                        countm[roi_id[aa],roi_id[bb]] = countm[roi_id[aa],roi_id[bb]] + 1
    return countm

def tworoi_passingrois_matinput(label_img,N,groupsl,roia,roib,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)

    tmp_streamlines = groupsl[roia-1,roib-1]
    [tmp1,tmp2,Nfibers] = tmp_streamlines.shape
    passroi_2nodes = []
    for n in range(0,Nfibers):
        tmp = tmp_streamlines[:,:,n]
        sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
        slpoints = _to_voxel_coordinates(sl, lin_T, offset)
        ii, jj, kk = slpoints.T

        newlabel_img = label_img[ii,jj,kk]
        passroi_2nodes.append(newlabel_img)

    return passroi_2nodes

def rois_passingrois_matinput(label_img,N,groupsl,voxel_size=None,affine=None):
    # input: label_img: image of labels, index from 0 - N
    #        N: # of ROIs
    #        groupsl: streamlines connecting ith and jth ROIs, groupsl[i,j]
    #        voxel_size, affine: some parameters to transfore streamlines to voxel
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    passroicm = defaultdict(list)

    for i in range(0,N-1):
        for j in range(i+1,N):
            tmp_streamlines = groupsl[i,j]
            [tmp1,tmp2,Nfibers] = tmp_streamlines.shape
            passroicm_2nodes = []
            for n in range(0,Nfibers):
                tmp = tmp_streamlines[:,:,n]
                sl = tmp.transpose() #flip the matrix to get the right input for extracting fa;
                slpoints = _to_voxel_coordinates(sl, lin_T, offset)
                ii, jj, kk = slpoints.T

                newlabel_img = label_img[ii,jj,kk]
                passroicm_2nodes.append(newlabel_img)
            passroicm[i,j].append(passroicm_2nodes)

    return passroicm