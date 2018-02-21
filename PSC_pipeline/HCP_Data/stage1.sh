# Stage 1 of PSC pipeline. In this step, we process dwi and t1 image to 
# get them ready for connectome analysis


#remove all previous results
rm -r diffusion
rm -r structural
rm -r streamlines
rm -r diffution
rm -r connectome

mkdir diffusion/

#translate data.nii.gz, nodif_brain_mask.nii.gz to [+1 +2 +3 +4] space 
#this command is form Mrtrix3.0
mrconvert data.nii.gz -stride 1,2,3,4  diffusion/data.nii.gz -force
mrconvert nodif_brain_mask.nii.gz -stride 1,2,3 diffusion/mask.nii.gz -force

#translate bval bvec to the right space [+1 +2 +3 +4] 
scil_convert_gradient_fsl_to_mrtrix.py -bval bvals -bvec bvecs -e diffusion/encoding.b 
flip_grad_mrtrix.py -b diffusion/encoding.b -d x -o diffusion/encoding_x.b
convert_grad_mrtrix2fsl.py -e diffusion/encoding_x.b -bval diffusion/bvals_x -bvec diffusion/bvecs_x

#processe dMRI image
cd diffusion
#crop the diffusion MRI
auto-crop_volume.py mask.nii.gz mask_crop.nii.gz --coords_out box.pkl -f
apply_crop_bb.py data.nii.gz dwi_all.nii.gz box.pkl -f

#compute dti metrics
compute_dti_metrics.py dwi_all.nii.gz bvals_x bvecs_x --mask mask_crop.nii.gz -f
   
#computer odf metric
compute_fodf.py dwi_all.nii.gz bvals_x bvecs_x --mask mask_crop.nii.gz -f 

# clean files that we don't need
rm data.nii.gz # keep dwi_all.nii.gz

cd ..


# processe t1 image
mkdir structural

mrconvert T1w_acpc_dc_restore_1.25.nii.gz -stride 1,2,3 structural/t1.nii.gz -force

#extract the brain
mrcalc structural/t1.nii.gz diffusion/mask.nii.gz -mult structural/t1_brain.nii.gz -force

#denoising
run_nlmeans.py structural/t1_brain.nii.gz structural/t1_brain_nlm.nii.gz -N 1 --noise_est basic -f
#crop
apply_crop_bb.py structural/t1_brain_nlm.nii.gz structural/t1_brain_crop.nii.gz diffusion/box.pkl -f
#segment
cd structural
fast -t 1 -n 3 -H 0.1 -I 6 -l 20.0 -g -o t1_brain_crop.nii.gz t1_brain_crop.nii.gz
mv t1_brain_crop_seg_2.nii.gz mask_wm.nii.gz
mv t1_brain_crop_seg_1.nii.gz mask_gm.nii.gz
mv t1_brain_crop_seg_0.nii.gz mask_csf.nii.gz
mv t1_brain_crop_pve_2.nii.gz map_wm.nii.gz
mv t1_brain_crop_pve_1.nii.gz map_gm.nii.gz
mv t1_brain_crop_pve_0.nii.gz map_csf.nii.gz
compute_pft_maps.py map_wm.nii.gz map_gm.nii.gz map_csf.nii.gz -f
count_non_zero_voxels.py interface.nii.gz interface_count.txt


# processe the label data 
mrconvert ../wmparc.nii.gz -stride 1,2,3 wmparc_0.7.nii.gz -force
resample.py wmparc_0.7.nii.gz wmparc.1.25.nii.gz --ref t1_brain.nii.gz -f --interp nn
#mrtransform -template t1_brain.nii.gz -interp nearest -datatype int32 wmparc_0.7.nii.gz mr_wmparc.1.25.nii.gz -force

mrconvert ../aparc.a2009s+aseg.nii.gz -stride 1,2,3 aparc.a2009s+aseg_0.7.nii.gz -force
resample.py aparc.a2009s+aseg_0.7.nii.gz aparc.a2009s+aseg.1.25.nii.gz --ref t1_brain.nii.gz -f --interp nn

#crop
apply_crop_bb.py aparc.a2009s+aseg.1.25.nii.gz aparc.a2009s+aseg_crop.nii.gz ../diffusion/box.pkl -f 
apply_crop_bb.py wmparc.1.25.nii.gz wmparc_crop.nii.gz ../diffusion/box.pkl -f

cd ..


##......... this step is not necessary  ..........#
#cd diffusion
#compute_qball_metrics.py dwi_all.nii.gz bvals_x bvecs_x --mask mask_crop.nii.gz --gfa gfa.nii.gz --sh qball_csa.nii.gz -f
