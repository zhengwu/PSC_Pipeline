# run free surfer for T1 image
# t1 image name is T1_dti_final, should be in the structural folder

recon-all -subjid football_sub002 -i T1_dti_final.nii.gz -all

# get to the subject specific folder;
mri_convert -rl rawavg.mgz -rt nearest wmparc.mgz wmparc_in_rawavg.mgz
mri_convert -rl rawavg.mgz -rt nearest aparc.a2009s+aseg.mgz aparc.a2009s+aseg_in_rawavg.mgz

#after runing freesurfer, move the volume pacellation to the data folder
# copy the following files to the subject's main folder
# brainmask.mgz