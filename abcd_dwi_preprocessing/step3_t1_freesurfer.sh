# run freesurfer for t1 image
SRUN=srun

cd ses-baselineYear1Arm1/dwi_psc_connectome
cd structure

export SUBJECTS_DIR=$(pwd)

# t1 image name is T1_dti_final, should be in the "structure" folder
$SRUN recon-all -subjid t1_freesurfer -i t1_wholebrain_warped.nii.gz -all



# get to the subject specific folder;
cd t1_freesurfer
cd mri
mri_convert -rl rawavg.mgz -rt nearest wmparc.mgz wmparc_in_rawavg.mgz
mri_convert -rl rawavg.mgz -rt nearest aparc.a2009s+aseg.mgz aparc.a2009s+aseg_in_rawavg.mgz

#after runing freesurfer, move the volume pacellation to the data folder
# copy the following files to the subject's main folder

cp wmparc_in_rawavg.mgz ../../wmparc_in_rawavg.mgz
cp aparc.a2009s+aseg_in_rawavg.mgz ../../aparc.a2009s+aseg_in_rawavg.mgz

############### processe the label data
cd ..
cd ..
mrconvert wmparc_in_rawavg.mgz wmparc_in_rawavg.nii.gz -force
mrconvert wmparc_in_rawavg.nii.gz -stride 1,2,3 wmparc.nii.gz -force


#mrtransform -template t1_brain.nii.gz -interp nearest -datatype int32 wmparc_0.7.nii.gz mr_wmparc.1.25.nii.gz -force
mrconvert aparc.a2009s+aseg_in_rawavg.mgz aparc.a2009s+aseg_in_rawavg.nii.gz -force
mrconvert aparc.a2009s+aseg_in_rawavg.nii.gz -stride 1,2,3 aparc.a2009s+aseg.nii.gz -force


#align the pacellation data to the right space since freesurfer resampled the data;
antsRegistrationSyNQuick.sh -d 3 -f t1_warped.nii.gz -m t1_wholebrain_warped.nii.gz -o ForParcel_ -t r   
# This just does a rigid registration, really quick, 
# and just reslices wmparc to make it have the same affine and size as t1_warped.nii.gz
antsApplyTransforms -d 3 -i wmparc.nii.gz -r t1_warped.nii.gz -o wmparc_warped_label.nii.gz -n MultiLabel -t ForParcel_0GenericAffine.mat
antsApplyTransforms -d 3 -i aparc.a2009s+aseg.nii.gz -r t1_warped.nii.gz -o aparc.a2009s+aseg_warped_label.nii.gz -n MultiLabel -t ForParcel_0GenericAffine.mat

#get out the folder
cd ..
cd ..
cd ..