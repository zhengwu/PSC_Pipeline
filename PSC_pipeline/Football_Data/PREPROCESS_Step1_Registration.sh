# This script is doing two things:
# 1. prepare data for tractography
# 2. register t1 to diffusion (d0 and fa) space
# 3. segment t1 image

# To run this script, you need to set up
# 1. Zhengwu's latest Scilpy script
# 2. ANTs - the latest one (ANTs2)
# 3. FSL - will use flirt to perform registration

# Author: Zhengwu Zhang
# Date: Mar. 23, 2017

#remove all previous results
rm -r diffusion
rm -r structural
rm -r streamlines
rm -r diffution
rm -r connectome
rm -r registration


# make registration folder to hold all intermedia files for registration
mkdir registration/
mkdir structural/

# files to hold all intermedia results for diffusion analysis
mkdir diffusion/

# here we assume that the data.nii.gz has been preporcessed by FSL's pipeline - ie eddy correction and susceptibility-induced distortion correction
mrconvert data.nii.gz -stride 1,2,3,4  diffusion/data.nii.gz -force

#translate bval bvec to the right space [+1 +2 +3 +4] 
scil_convert_gradient_fsl_to_mrtrix.py bvals bvecs diffusion/encoding.b 
scil_flip_grad.py --mrtrix diffusion/encoding.b diffusion/encoding_x.b x 
scil_convert_gradient_mrtrix_to_fsl.py diffusion/encoding_x.b diffusion/bvals_x diffusion/bvecs_x

#resample t1 image into 1x1x1 
scil_resample_volume.py T1.nii.gz structural/T1_1x1x1.nii.gz --resolution 1 -f

#denoise the t1 image 
scil_run_nlmeans.py structural/T1_1x1x1.nii.gz structural/T1_1x1x1_denoised.nii.gz 1 --noise_est basic -f

#resample the dti image into 1x1x1 
scil_resample_volume.py diffusion/data.nii.gz diffusion/data_1x1x1.nii.gz --resolution 1 -f


####### registration between dti and t1 ##########
####### initial registration using FSL
flirt -in structural/T1_1x1x1_denoised.nii.gz -ref diffusion/data_1x1x1.nii.gz -out registration/T1_dti_1.nii.gz -omat registration/T1_dti_1.mat -bins 256 -cost mutualinfo -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp sinc -sincwidth 7 -sincwindow hanning


## improved the registration
# from DTI data, extract b0
bet diffusion/data_1x1x1.nii.gz mask.nii.gz -R -m -f 0.25
mv mask.nii.gz b0_brain.nii.gz


# using the same mask, extract t1 image
fslmaths registration/T1_dti_1.nii.gz -mul mask_mask.nii.gz registration/mask_dt1_t1.nii


#calculate the fa map for dti data
cd diffusion
scil_compute_dti_metrics.py data_1x1x1.nii.gz bvals_x bvecs_x --mask ../mask_mask.nii.gz -f \
--not_all --fa fa.nii.gz --tensor tensor.nii.gz

#now we have fa, do registration using fa, b0 data together
cd ..

# Use ANTS to improve the registration. This does not work on a non-masked version.
antsRegistration -d 3 -m MI[b0_brain.nii.gz,registration/mask_dt1_t1.nii.gz,1,32,Regular,0.25] \
-m MI[diffusion/fa.nii.gz,registration/mask_dt1_t1.nii.gz,1,4] \
-c [1000x500x250x0,1e-7,5] -t affine[0.1] -f 8x4x2x1 -s 4x2x1x0 -u 1 -o Antsaffine

antsApplyTransforms -d 3 -i registration/T1_dti_1.nii.gz -o T1_dti_final.nii.gz \
 -r b0_brain.nii.gz -t Antsaffine0GenericAffine.mat -n Linear


# rename the mask;
bet T1_dti_final.nii.gz t1_brain.nii.gz -m -R -B -f 0.2
mv t1_brain_mask.nii.gz nodif_brain_mask.nii.gz
