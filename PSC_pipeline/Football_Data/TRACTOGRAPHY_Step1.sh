###### this code is developed for processing PNC data
# author: Zhengwu Zhang
# date: Mar. 23, 2017

# To run this script, you need to set up
# 1. Zhengwu's latest Scilpy script
# 2. MRtrix 3

#PNC dataset fibers extraction

# improve the brain segmentation (from the freesufer result)
mrconvert brainmask.mgz brainmask.nii.gz -force
mrconvert brainmask.nii.gz -stride 1,2,3 brainmask.nii.gz -force
mrtransform -template T1_dti_final.nii.gz -interp linear -datatype int32 brainmask.nii.gz mr_brainmask.nii.gz -force

fslmaths mr_brainmask.nii.gz -bin t1_brain_mask.nii.gz 
fslmaths t1_brain_mask.nii.gz -add nodif_brain_mask.nii.gz merged_brain_mask.nii.gz
fslmaths merged_brain_mask.nii.gz -bin merged_brain_mask.nii.gz 

# rename the mask;
mv merged_brain_mask.nii.gz nodif_brain_mask.nii.gz

#mv final t1 to structural data
mv T1_dti_final.nii.gz structural/T1_dti_final.nii.gz
fslmaths structural/T1_dti_final.nii.gz -mul nodif_brain_mask.nii.gz structural/t1_brain.nii.gz
#mv mr_brainmask.nii.gz structural/t1_brain.nii.gz


#processe dMRI image
cd diffusion

# crop the diffusion data
#auto-crop_volume.py ../b0_brain.nii.gz ../b0_crop.nii.gz --coords_out box.pkl -f
#apply_crop_bb.py data_1x1x1.nii.gz dwi_all.nii.gz box.pkl -f

mrconvert ../nodif_brain_mask.nii.gz -stride 1,2,3 mask.nii.gz -force


#apply_crop_bb.py mask.nii.gz mask_crop.nii.gz box.pkl -f

######## compute dti metrics
#scil_compute_dti_metrics.py data_1x1x1.nii.gz bvals_x bvecs_x --mask mask_crop.nii.gz -f
scil_compute_dti_metrics.py data_1x1x1.nii.gz bvals_x bvecs_x --mask mask.nii.gz -f

########## computer odf metric, denoising the data
# this step takes about 1~1.5 hours
#scil_run_nlmeans.py dwi_all.nii.gz dwi_crop_rnlm.nii.gz 1 \
#    --mask mask_crop.nii.gz --noise_est basic

scil_run_nlmeans.py data_1x1x1.nii.gz dwi_rnlm.nii.gz 1 \
    --mask mask.nii.gz --noise_est basic


cd ..


######### processe t1 image - T1_dti_final.nii.gz in structural 

#extract the brain
#mrcalc structural/T1_dti_final.nii.gz diffusion/mask.nii.gz -mult structural/t1_brain.nii.gz -force

#denoising
#run_nlmeans.py structural/t1_brain.nii.gz structural/t1_brain_nlm.nii.gz -N 1 --noise_est basic -f
#crop
#apply_crop_bb.py structural/t1_brain.nii.gz structural/t1_brain_crop.nii.gz diffusion/box.pkl -f

#segment
cd structural
fast -t 1 -n 3 -H 0.1 -I 6 -l 20.0 -g -o t1_brain.nii.gz t1_brain.nii.gz
mv t1_brain_seg_2.nii.gz mask_wm.nii.gz
mv t1_brain_seg_1.nii.gz mask_gm.nii.gz
mv t1_brain_seg_0.nii.gz mask_csf.nii.gz
mv t1_brain_pve_2.nii.gz map_wm.nii.gz
mv t1_brain_pve_1.nii.gz map_gm.nii.gz
mv t1_brain_pve_0.nii.gz map_csf.nii.gz

#compute_pft_maps.py map_wm.nii.gz map_gm.nii.gz map_csf.nii.gz -f
scil_compute_maps_for_particle_filter_tracking.py map_wm.nii.gz map_gm.nii.gz map_csf.nii.gz -f
scil_count_non_zero_voxels.py interface.nii.gz interface_count.txt


############### processe the label data 
mrconvert ../wmparc_in_rawavg.mgz ../wmparc_in_rawavg.nii.gz -force
mrconvert ../wmparc_in_rawavg.nii.gz -stride 1,2,3 wmparc.nii.gz -force
#apply_crop_bb.py mr_wmparc.nii.gz wmparc_crop.nii.gz ../diffusion/box.pkl -f
#align the pacellation data to the right space since freesurfer resampled the data;

#mrtransform -template t1_brain.nii.gz -interp nearest -datatype int32 wmparc_0.7.nii.gz mr_wmparc.1.25.nii.gz -force
##!!!!!!!!!!!!!! needs revision
mrconvert ../aparc.a2009s+aseg_in_rawavg.mgz ../aparc.a2009s+aseg_in_rawavg.nii.gz -force
mrconvert ../aparc.a2009s+aseg_in_rawavg.nii.gz -stride 1,2,3 aparc.a2009s+aseg.nii.gz -force

#mrconvert ../aparc.a2009s+aseg.mgz ../aparc.a2009s+aseg.nii.gz -force
#mrconvert ../aparc.a2009s+aseg.nii.gz -stride 1,2,3 aparc.a2009s+aseg.nii.gz -force
#mrtransform -template t1_brain.nii.gz -interp nearest -datatype int32 aparc.a2009s+aseg.nii.gz mr_aparc.a2009s+aseg.nii.gz -force
#apply_crop_bb.py mr_aparc.a2009s+aseg.nii.gz aparc.a2009s+aseg_crop.nii.gz ../diffusion/box.pkl -f 

cd ..


############### calculate fodf
cd diffusion
#scil_compute_fodf.py dwi_crop_rnlm.nii.gz bvals_x bvecs_x --sh_order 6 --mask mask_crop.nii.gz -f --frf 15,3,3 --not_all --fodf fodf.nii.gz --peaks peaks.nii.gz 
scil_compute_fodf.py dwi_rnlm.nii.gz bvals_x bvecs_x --sh_order 6 --mask mask.nii.gz --mask_wm ../structural/mask_wm.nii.gz --processes 1 -f --frf 15,3,3 --not_all --fodf fodf.nii.gz --peaks peaks.nii.gz 
#scil_compute_fodf.py dwi_rnlm.nii.gz bvals_x bvecs_x --basis mrtrix --processes 1 --sh_order 6 --mask mask.nii.gz -f --frf 15,3,3 --not_all --fodf fodf_mrtrix.nii.gz 


# clean large files that we don't need
rm data.nii.gz # keep dwi_all.nii.gz
rm data_1x1x1.nii.gz
rm dwi_rnlm.nii.gz

cd ..