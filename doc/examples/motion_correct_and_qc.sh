# example script doing DWI raw data quality check while computing DTI

dwi_animation.py dwi_2x2x2_nlm.nii -rs &
# I like dwi_animation better, scales better 

################################################
# Run EDDY on the denoised image, 2x2x2 space
################################################
# Prepare txt files for the EDDY. 
# The final thing we need to do is to create an index file that tells eddy which line/of the lines in the acqparams.txt file that are relevant for the data passed into eddy. In this case all the volumes in dwi_2x2x2_nlm.nii are acquired A>>P which means that the first line of acqparams.txt describes the acquisition for all the volume. We specify that by passing a text file with as many ones as there are volumes in data.nii.gz. One way of creating such a file would be to type the following commands
indx=""
for ((i=1; i<=65; i+=1)); do indx="$indx 1"; done
echo $indx > index.txt
echo "0 1 0 0.051" > acqparams.txt
# Here, 0.051 is a dummy number. When you have only a A>>P it is not important to be exact
# Here, 65 is the 4th dimension of the dwi_2x2x2_nlm.nii dataset

eddy --imain=dwi_2x2x2_nlm.nii --mask=b0_2x2x2_brain_mask.nii.gz --index=index.txt --acqp=acqparams.txt --bvecs=bvec --bvals=bval --out=dwi_2x2x2_nlm_eddy.nii.gz  

dwi_animation.py dwi_2x2x2_nlm_eddy.nii.gz -rs &

###############
# rotate bvecs
###############
path="$HOME/Research/Source/scilpy/scripts/"
python $path/rotate_bvecs_eddyparameters.py bvec dwi_2x2x2_nlm_eddy.nii.gz.eddy_parameters False motion_max_translation.txt

###########################################################
# computed DTI metrics with rotated bvecs and rotated DWIs
###########################################################
compute_dti_metrics.py dwi_2x2x2_nlm_eddy.nii.gz bval bvec_eddy --mask b0_2x2x2_brain_mask.nii.gz -f 

# count number of physically implausible voxels in the white matter mask
fslmaths b0_2x2x2_brain_mask.nii.gz -ero -kernel sphere 5 b0_2x2x2_brain_mask_eroded.nii.gz 
mrmult physically_implausible_signals_mask.nii.gz b0_2x2x2_brain_mask_eroded.nii.gz physically_implausible_signals_mask_eroded.nii.gz 
resample.py --interp nn physically_implausible_signals_mask_eroded.nii.gz  physically_implausible_signals_mask_eroded_1x1x1.nii.gz  --ref b0_wm_mask_ants.nii 
mrmult physically_implausible_signals_mask_eroded_1x1x1.nii.gz b0_wm_mask_ants.nii physically_implausible_voxels_1x1x1.nii.gz  
count_non_zero_voxels.py physically_implausible_signals_mask_eroded.nii.gz physically_implausible_voxels.txt

#############
# Reporting
# 
# To do:
# 1) axial, sagittal, coronal maps Biospective-style (i.e. little mosaic) of :
#   i) residual.nii.gz
#   ii) pulsation_and_misalignment_std_dwi.nii.gz
#   iii) physically_implausible_signals_mask
#   iv) bvecs_eddy.png + motion_max_translation.txt in the report
#
# 2) What do we do with physically implausible signals?
#     - Nothing for now. We just count and plot at the end over all subjects.
#     - Same for motion_max_translation
#
