# Instructions for the surgery pipeline

## Setup for scilpy surgery scripts
* The scilpy/surgery_scripts and scilpy/scripts folders need to be added to your $PATH.
* This script supports mrtrix 2 and 3, but make sure to check log file if the script crash quickly at the beginning. One of the very first step is mrconvert.
* The script supports FSL-5.0 (with and without the alias) for now.
* Ants is required in most cases.
* The advanced BET can use AntsBrainExtraction.sh which is superior to FSL, to be able to use it you need to download this template (URL) and copy the template folder in the scilpy/data/ folder

## How to call the surgery preprocessing script
Note that the script needs the right input (order, file format, datatype) and cannot handle flipped datasets (T1 & B0).

With a folder filled with dicom files or a DICOMDIR file
```
surgery_diffusion_preprocessing.sh raw_dicom_folder/
```
With a previous launch parameter file
```
surgery_diffusion_preprocessing.sh logdir/main_parameters.txt
```
With pre-converted nifti files (the user still need to start from step 1)
```
surgery_diffusion_preprocessing.sh dwi.nii.gz encoding.b t1.nii.gz
```
This script encompasses all the major preprocessing steps for our diffusion pipeline. Various questions are asked to the user to adapt the complexity of the pipeline and it can range from 15 minutes to more than 1 hour. Rough estimations of the time needed are available for each step in the script.

## How to call the generate streamlines script
This script is designed to use ther the output of the preprocessing script, but the code can be easily modified for other scenarios. First you need to change directory to the *final_data/* folder created at the end of the preprocessing script. Then you can call the script and specify if you want WM or FA seeding (for WM you need to have selected the tissues segmentation option). You also can choose if you want deterministic or probabilistic streamlines and finally the number of seed you want to generate (--nt is used)
```
surgery_streamlines_generation.sh {fa, wm} {prob, det} ######
```
This script produces a basic tractography after the preprocessing and helps to do a quick QA. For more specific needs the script can be modified, for example more parameters can be added.

## How to call the streamlines deformation wrapper
This script helps with with nonlinear deformation of streamlines, as it is difficult to know which scripts to call and to reduce space-related problems or subtlety of the Ants warp file format.

This script requires 5 arguments, obtained in a similar way to the one created by the preprocessing script.

* The .mat file created by Ants linear transformation.
* The **inverse** warp field created by Ants.
* The .trk file you want to warp.
* The final reference .nii.gz or .trk, the output file will have the same dimensions, voxel sizes and affine in its header
* This arguments should usually be 'y', if you originally computed a transformation to diffusion space from another space. **BUT** if you originally computed a transformation to another space from diffusion, you need to use 'n' as a fifth argument and use the **direct** warp field as the second argument.
* The final output filename of the script

```
surgery_streamlines_deformation.sh Affine.mat InverseWarp.nii.gz whole_brain.trk t1.nii.gz y whole_brain_warp.trk
```
Ants affine matrix are often named *output0GenericAffine.mat* and the warp field are named *output1Warp.nii.gz* and *output1InverseWarp.nii.gz*.
This script is also easy to modify to fit your need, for example if you want to perform only a linear transformation or if you need to perform a sequence of transformations one after the other.

## How to call the personalized pipeline script
This script is a wrapper for a fixed pipeline made for a collaborator. The degree of freedom is reduced by fixing parameters and automatically generating streamlines and moving them back onto the original high resolution T1. This script can take more than 1h for the preprocessing, 6h for the streamlines generation and 15 minutes for streamlines deformation. Use with caution, to verify if your data is compatible with our pipeline you should run a more simple version first and do Q&A before lauching this more advanced (and slow) version.
```
emmanuel_pipeline.sh dwi_center.nii.gz encoding.b t1_hr.nii.gz
```

## Tips and precautions
* Resampling peaks to T1 resolution enhances the quality of visualization when compared to diffusion space.
* If a step fails or if you want to change a decision about one of the steps, you can kill the process, recall the script with *main_parameters.txt* (from the most recent log directory) and restart it from the same step it was in when you killed it. This will not change the decision you made, but you can easily change the caracters in the parameters.
* Calling the scripts without any argument will display a warning with the informations in red.
* If your T1 has a resolution above 1mm isotropic, we advise you to use the high resolution option to speed up computation. Or to inspire yourself from the *emmanuel_pipeline.sh*. Resampling DWI to a high resolution will likely cause a crash by taking all of your RAM.
* The streamlines generation script will only work from within the final_data/ folder, filenames are fixed for simplification.