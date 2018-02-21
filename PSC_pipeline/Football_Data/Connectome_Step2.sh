# cd to the data folder


###### extract the streamline connectivity matrix for both cortical and subcortical 
cd connectome

#set scilpy's path
export SCIP_PATH=/Users/zzheng6/Sofeware/Scilpy

# extract the connectivity matrix, get the dilation of images for Desikan
extraction_streamlinesccm.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/wmparc.nii.gz $SCIP_PATH/scilpy/connectome/Desikan_ROI.txt $SCIP_PATH/scilpy/connectome/FreeSurferColorLUT.txt Football 30 240 2 2 4 desikan

echo "extraction of cortical connection using desikan atls is sucessful \n"

# extract the connectivity matrix for Destreoux
 extraction_streamlinescm.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/aparc.a2009s+aseg_crop.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Destrieux_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt HCP 30 240 2 2 4 destrieux
echo "extraction of cortical connection using destreoux atls is sucessful \n"


# extract the Subcortical_ROI.txt
extraction_streamline_withsubcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/aparc.a2009s+aseg_crop.nii.gz HCP_destrieux_dilated_labels.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Subcortical_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt HCP 30 240 4 0 destrieux
extraction_streamline_withsubcortical.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/wmparc_crop.nii.gz HCP_desikan_dilated_labels.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Subcortical_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt HCP 30 240 4 0 desikan
echo "extraction of subcortical connection using desikan and destreoux atls is sucessful \n"



########## using matlab to process the streamlines
# we need to use matlab to generate the code at each folder first
matlab -nodisplay -nosplash -singleCompThread  -r process_pythonoutput -logfile outputmatlab

echo "matlab processing of streamline connectivity is successful \n"


############### extract features
connectivity_featurematrix_extraction.py HCP_destrieux_cm_processed_streamlines_100_p.mat ../streamlines/full_interface_prob_pft_invcoord.trk HCP_destrieux_dilated_labels.nii.gz ../diffusion/fa.nii.gz ../diffusion/md.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Destrieux_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt destrieux 
mkdir cm_matrix 
mv HCP_destrieux_cm_processed_c* cm_matrix 
mv HCP_destrieux_cm_processed_f* cm_matrix 
mv HCP_destrieux_cm_processed_v* cm_matrix 
mv HCP_destrieux_cm_processed_m* cm_matrix


connectivity_featurematrix_extraction.py HCP_desikan_cm_processed_streamlines_100_p.mat ../streamlines/full_interface_prob_pft_invcoord.trk HCP_desikan_dilated_labels.nii.gz ../diffusion/fa.nii.gz ../diffusion/md.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Desikan_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt desikan 
mkdir cm_matrix 
mv HCP_desikan_cm_processed_c* cm_matrix 
mv HCP_desikan_cm_processed_f* cm_matrix 
mv HCP_desikan_cm_processed_v* cm_matrix 
mv HCP_desikan_cm_processed_m* cm_matrix

echo "extraction of connectome feature is done... \n"


############### extract the FA values
connectivity_famat_extraction.py HCP_destrieux_cm_processed_streamlines_100_p.mat ../streamlines/full_interface_prob_pft_invcoord.trk HCP_destrieux_dilated_labels.nii.gz ../diffusion/fa.nii.gz ../diffusion/md.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Destrieux_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt destrieux 
connectivity_famat_extraction.py HCP_desikan_cm_processed_streamlines_100_p.mat ../streamlines/full_interface_prob_pft_invcoord.trk HCP_desikan_dilated_labels.nii.gz ../diffusion/fa.nii.gz ../diffusion/md.nii.gz /Volumes/TestDataSet/PNC/PSCPipeline/Desikan_ROI.txt /Volumes/TestDataSet/PNC/PSCPipeline/FreeSurferColorLUT.txt desikan 

# we need to use matlab to generate the code at each folder first
matlab -nodisplay -nosplash -singleCompThread  -r process_fa_pythonoutput -logfile outputmatlab_fa

echo "processing of FA is done... \n"

############### multiple resolution
#extract_streamlinecm_multiresolution.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/aparc.a2009s+aseg_crop.nii.gz HCP_destrieux_dilated_labels.nii.gz  \
#HCP 20 240 20 destrieux

#extract_streamlinecm_multiresolution.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/wmparc_crop.nii.gz HCP_desikan_dilated_labels.nii.gz \
#HCP 20 240 20 desikan

#echo "processing of multiresolution is done... \n"

#remove some files to release the space
rm HCP_destrieux_cm_streamlines.mat
rm HCP_destrieux_cm_streamlines_100_p.mat 
rm HCP_desikan_cm_streamlines.mat 
rm HCP_desikan_cm_streamlines_100_p.mat

fiberlength_distribution_extraction.py ../streamlines/full_interface_prob_pft_invcoord.trk ../structural/t1_brain_crop.nii.gz HCP_  

