
####################set PSC's path###########
mkdir connectome

#### change to your own path
#export PSC_PATH=/home/zzhang87/dwi_pipeline/PSC
#export PSC_PATH=/home/zzhang6/Software/PSC_Pipeline
export PSC_PATH=/Users/zzheng6/Sofeware/PSC_Pipeline
#SRUN=srun
SRUN=
DatasetNAME=UKBIOBANK

# cd to the data folder
cd connectome

###### extract the streamline connectivity matrix for both cortical and subcortical 

#desikan atlas
# extract connectivity matrices, get the dilation of images for Desikan
$SRUN extraction_sccm_withfeatures_cortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/dti/fa.nii.gz ../diffusion/dti/md.nii.gz ../structure/wmparc.nii.gz $PSC_PATH/connectome/Desikan_ROI.txt $PSC_PATH/connectome/FreeSurferColorLUT.txt $DatasetNAME 20 240 1 2 0.8 desikan
$SRUN extraction_sccm_withfeatures_subcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/dti/fa.nii.gz ../diffusion/dti/md.nii.gz ../structure/wmparc.nii.gz HCP_desikan_dilated_labels.nii.gz  $PSC_PATH/connectome/Subcortical_ROI.txt $PSC_PATH/connectome/FreeSurferColorLUT.txt $DatasetNAME 20 240 0.8 0 desikan


# Destreoux
$SRUN extraction_sccm_withfeatures_cortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/dti/fa.nii.gz ../diffusion/dti/md.nii.gz ../structure/aparc.a2009s+aseg.nii.gz $PSC_PATH/connectome/Destrieux_ROI.txt $PSC_PATH/connectome/FreeSurferColorLUT.txt $DatasetNAME 20 240 1 2 0.8 destrieux
$SRUN extraction_sccm_withfeatures_subcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../diffusion/dti/fa.nii.gz ../diffusion/dti/md.nii.gz ../structure/aparc.a2009s+aseg.nii.gz HCP_destrieux_dilated_labels.nii.gz  $PSC_PATH/connectome/Subcortical_ROI.txt $PSC_PATH/connectome/FreeSurferColorLUT.txt $DatasetNAME 20 240 0.8 0 destrieux


#rm HCP_destrieux_partbrain_subcort_cm_processed_sfa_100.mat
#rm HCP_destrieux_partbrain_cm_processed_smd_100.mat
#rm HCP_destrieux_partbrain_subcort_cm_streamlines.mat

#rm HCP_desikan_partbrain_subcort_cm_processed_sfa_100.mat
#rm HCP_desikan_partbrain_cm_processed_smd_100.mat]
#rm HCP_desikan_partbrain_subcort_cm_streamlines.mat