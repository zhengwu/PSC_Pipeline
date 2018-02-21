#!/bin/bash

echo 'Fetching data needed to run the script...'
wget https://www.dropbox.com/s/9esgdpoya7y1n6x/local_reconstruction.zip?dl=0 
unzip local_recon*
mv local_reconstruction/* .
rm -rf local_reconstruction/ 

echo
echo 'Starting processing...'
echo 'Extracting brain mask...'
bet dwi_1x1x1.nii.gz mask.nii.gz -m -R -f 0.15

echo 'Compute DTI metrics...'
scil_compute_dti_metrics.py dwi_1x1x1.nii.gz bvals bvecs --mask mask_mask.nii.gz -f

echo 'Compute qball GFA metric...'
scil_compute_qball_metrics.py dwi_1x1x1.nii.gz bvals bvecs --mask mask_mask.nii.gz \
    --not_all --gfa gfa.nii.gz -f

echo 'Compute fODFs...'
scil_compute_fodf.py dwi_1x1x1.nii.gz bvals bvecs --mask mask_mask.nii.gz --frf 15,3,3 -f
mkdir -p metrics qa
mkdir -p local_models
mv ad* rd* md* mode* ga* fa* gfa* rgb* *evals* tensor_norm* metrics/
mv fodf* peak* tensor* local_models/
mv mask_* dti_* phys* pul* qa/
scil_count_non_zero_voxels.py qa/physically_implausible_signals_mask.nii.gz \
    > qa/physically_implausible_number_of_voxels.txt

echo
echo 'Done processing.'
echo 'Please have a look at the comments in the script to see what you can do next with your data.'

# 1) Check metrics/FA and RGB (color FA)
#     Cingulum -> green
#     CST      -> blue
#     CC       -> red
#
# 2) Check qa/residuals.nii.gz 
#     - should be quite uniform
#     - should should highlight artefacts in the data
#
# 3) qa/physically_implausible_voxels.nii.gz
#     - should not be inside the white matter
#
# 4) Check local_models/tensors.nii.gz in FiberNavigator
#     Principal direction needs to be 
#     Cingulum -> Front-Back
#     CST      -> Up-Down
#     CC       -> Left-Right
#     No weird flips?
#
# 5) Check local_models/peaks.nii.gz in FiberNavigator 
#      - File -> Open anatomy as peaks
#      - Same check as 2)
#      - Check crossings in the centrum semiovale

# At this point, if 1) to 5) have passed and you have a good brain mask
# your tractography will be fine.

