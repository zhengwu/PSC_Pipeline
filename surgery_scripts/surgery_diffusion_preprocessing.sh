 #!/bin/bash
BLUE='\033[0;34m'
LRED='\033[1;31m'
NC='\033[0m'
ColorQuestion () { echo -n -e ${BLUE}${1}${NC}; }
ColorPrint () { echo -e ${BLUE}${1}${NC}; }

####################################################################
# All steps are ordered here, variables declaration & cleaning
####################################################################
function main ()
{
    # Declare global variables
    export logdir=logs_$(date "+%d-%m-%Y_%H-%M-%S")
    export bet_option denoise upsample1 upsample2 upsample3 registration \
        segmentation high_res_t1 step_start
    export was_denoised="" wasWarped=""
    export mrtrix_version
    version=$(mrinfo --version | grep 'mrinfo 0.*.*')
    [[ ${version} == *.2.* ]] && mrtrix_version=2 || mrtrix_version=3

    parsing_and_input $@

    # Convert the input to standardize themp
    if [ ${step_start} -le 1 ]
    then
        input2nii $@
    fi
    
    # Once the input are properly set, we save the parameters for restart
    echo ${bet_option} ${denoise} ${upsample1} ${upsample2} ${upsample3} \
    ${registration} ${segmentation} ${high_res_t1} >> ${logdir}/main_parameters.txt

    # Brain extraction tool, either FSL or BET (if the template is available)
    if [ ${step_start} -le 2 ]
    then
        ColorPrint "\nBET T1 AND DWI"
        brain_extraction_tool ${bet_option}
    fi

    # Standard cropping of the various volume
    if [ ${step_start} -le 3 ]
    then
        ColorPrint "\nDATA CROPPING AND B0 UPSAMPLING TO T1 SPACE"
        crop_data
    fi
    
    # Denoising
    if [ ${step_start} -le 4 ]
    then
        rm -f dwi_bet${was_denoised}16.nii.gz
        rm -f t1_bet${was_denoised}16.nii.gz
        if [ ${denoise} == 'y' ]
        then
            denoise_data
        fi
        
        # Must be done for registration in t1_res
        scil_resample_volume.py b0_bet_crop.nii.gz \
            b0_bet_crop${was_denoised}_t1_res.nii.gz --ref t1_bet_crop.nii.gz \
            --enforce_dimension -f \
            >>${logdir}/upsample_b0.txt 2>>${logdir}/upsample_b0.txt
        scil_resample_volume.py b0_bet_mask_crop.nii.gz \
            b0_bet_mask_crop_t1_res.nii.gz --ref t1_bet_crop.nii.gz \
            --enforce_dimension -f \
            >>${logdir}/upsample_b0.txt 2>>${logdir}/upsample_b0.txt
    fi

    if [ ${step_start} -le 5 ]
    then
        if [ ${upsample1} == 'y' ]
        then
            # Upsample to T1 space
            computation_t1_vizu_t1
        else
            # Stay in DWI space
            computation_diff
            if [ ${upsample2} == 'y' ]
            then
                # But upsample the peaks
                computation_diff_peaks_t1
            else
                # Must be done for registration in t1_res
                scil_resample_volume.py fa_diff_res.nii.gz fa_t1_res.nii.gz \
                    --ref t1_s.nii.gz --enforce_dimension -f \
                    >>${logdir}/resample_fa.txt 2>>${logdir}/resample_fa.txt
            fi
        fi
    fi

    if [ ${step_start} -le 6 ]
    then
        # Remove to avoid all possible confusion over the symlink.
        rm -f fa_sym.nii.gz b0_sym.nii.gz
        if [ ${upsample1} == 'y' ] || [ ${upsample2} == 'y' ] || [ ${upsample3} == 'y' ]
        then
            ln -s fa_t1_res.nii.gz fa_sym.nii.gz
            ln -s b0_bet_crop${was_denoised}_t1_res.nii.gz b0_sym.nii.gz
        else
            ln -s fa_diff_res.nii.gz fa_sym.nii.gz
            ln -s b0_bet_crop${was_denoised}.nii.gz b0_sym.nii.gz
        fi

        ants_registration ${registration}
    fi

    if [ ${step_start} -le 7 ] && [ ${segmentation} == 'y' ]
    then
        ColorPrint "\nWM/GM/CSF SEGMENTATION OF T1 (~10 min)"
        tissues_segmentation
    fi
    
    if [ ${step_start} -le 8 ]
    then
        ColorPrint "\nCREATING FINAL DATA FOLDER"
        copy_to_final_folder
    fi

    rm -f mrtrix-*.nii
    rm $(find ${logdir}/* -size 0 -print0 | xargs -0)
}

####################################################################
# Small subfunction to do a basic verification of the input
####################################################################
function verify_arguments ()
{
    missing=1
    if [ ${#} -eq 1 ] && [ -d ${1} ]; then
        missing=0
    fi

    if [ ${#} -eq 1 ] && [ -f ${1} ]; then
        missing=0
    fi

    if [ ${#} -eq 3 ]; then
        missing=0
    fi

    if [ ${missing} -eq 1 ]; then
        echo -e "${LRED}Missing some arguments."
        echo -e "  First argument : DICOM directory (folder)"
        echo -e "  OR"
        echo -e "  First argument : Main parameters logfile (file)"
        echo -e "  OR"
        echo -e "  First argument : Nifti file containing the DWI (*.nii, *.nii.gz)"
        echo -e "  Second argument : File containing the encoding scheme (*.b)"
        echo -e "  Third argument : Nifti file containing the T1 (*.nii, *.nii.gz)${NC}"
        exit 1
    fi
    mkdir -p ${logdir}
}

####################################################################
# Either read parameters from the file or use input from user
####################################################################
function parsing_and_input ()
{
    verify_arguments $@

    # Clean the folder of previous half-done mrtrix jobs
    rm -f mrtrix-*.nii

    # A restart needs to use the same parameters as before
    # If you know what you are doing, change the value in the file directly
    if [ ${#} -eq 1 ] && [ -f ${1} ]; then
        vector=$(cat ${1})
        bet_option=${vector:0:1}
        denoise=${vector:2:1}
        upsample1=${vector:4:1}
        upsample2=${vector:6:1}
        upsample3=${vector:8:1}
        registration=${vector:10:1}
        segmentation=${vector:12:1}
        high_res_t1=${vector:14:1}

        echo -e "${BLUE}Parameters were: ${NC}"
        echo -e "${BLUE}A - BET T1 robust to neck/shoulders: ${NC}" ${bet_option}
        echo -e "${BLUE}B - DENOISE THE DATA: ${NC}" ${denoise}
        echo -e "${BLUE}C - UPSAMPLE DWI TO T1 SPACE: ${NC}"  ${upsample1}
        echo -e "${BLUE}D - UPSAMPLE FODF FROM DIFF. SPACE TO T1 SPACE: ${NC}" ${upsample2}
        echo -e "${BLUE}E - UPSAMPLE PEAKS FROM FODF IN DIFF. SPACE TO T1 SPACE: ${NC}" ${upsample3}
        echo -e "${BLUE}F - NON LINEAR REGISTRATION (T1 to B0): ${NC}" ${registration}
        echo -e "${BLUE}G - WM/GM/CSF SEGMENTATION OF T1: ${NC}" ${segmentation}
        echo -e "${BLUE}G - HIGH RESOLUTION T1: ${NC}" ${high_res_t1}

        echo
        PS3="Select step to start from: "
        select word in "brain_extraction_tool" "cropping" "denoising" \
        "computation/visualization resampling" "registration" "segmentation" "copying"
        do
             step_start=$((${REPLY}+1))
             # Break, otherwise endless loop
             break
        done
    else
        # Personalized pipeline, options are meant to affect speed
        # to fit with the amount of time you have
        ColorQuestion "BET T1 robust to neck/shoulders (~25 min)? ('y' or 'n'): "
        read bet_option

        ColorQuestion "DENOISE THE DATA (~15 min)? ('y' or 'n'): "
        read denoise

        ColorQuestion "UPSAMPLE DWI TO T1 SPACE \n
          (slower processing but recommended if time)? ('y' or 'n'): "
        read upsample1

        if [ ${upsample1} == 'n' ]; then
            ColorQuestion "UPSAMPLE FODF FROM DIFF. SPACE TO T1 SPACE \n
               (peaks will be extracted from the upsampled fodf)? \n
               (fast - 'y' recommended)? ('y' or 'n'): "
            read upsample2
            if [ ${upsample2} == 'n' ]; then
                ColorQuestion "UPSAMPLE PEAKS FROM FODF IN DIFF. SPACE TO T1 SPACE \n
                   fast, cause block artefact - 'y' recommended)? ('y' or 'n'): "
                read upsample3
            else
                upsample3='n'
            fi
        else
            upsample2='n'
            upsample3='n'
        fi

        ColorQuestion "NON LINEAR REGISTRATION (T1 to B0) ('y' or 'n'): "
        read registration

        ColorQuestion "WM/GM/CSF SEGMENTATION OF T1
        (~10 min - not mandatory)? ('y' or 'n'): "
        read segmentation

        ColorQuestion "IS THE T1 HIGH RESOLUTION
        (will downsample the image to 1mm iso)? ('y' or 'n'): "
        read high_res_t1

        echo
        PS3="Select step to start from: "
        select word in "input2nii" "brain_extraction_tool" "cropping" "denoising" \
        "computation/visualization resampling" "registration" "segmentation" "copying"
        do
             step_start=${REPLY}
             # Break, otherwise endless loop
             break
        done
    fi

    # Instanciate variables default value
    if [ ${registration} == 'y' ]
    then
        was_warped="_warp"
    else
        was_warped="_linear"
    fi

    if [ ${denoise} == 'y' ]
    then
        was_denoised="_denoised"
    fi
}

####################################################################
# Convert DICOM to NIFTI using user input & standardize filename
####################################################################
function input2nii ()
{
    if [ ${mrtrix_version} -eq 3 ]; then
        arg_stride="-stride 1,2,3,4  -force"
        arg_grade="-export_grad_mrtrix"
    else
        arg_stride=""
        arg_grade="-grad"
        fi

    if [ ${#} -eq 1 ]; then
        rm -f dwi_s.nii.gz
        ColorPrint "\nCONVERSION OF DWI FILES"
        mrconvert ${1} dwi_s.nii.gz -datatype int16 ${arg_stride} -quiet

        rm -f encoding_s.b
        ColorPrint "\nCREATION OF ENCODING SCHEME FOR MRTRIX"
        mrinfo ${1} ${arg_grade} encoding_s.b -quiet

        rm -f t1_s.nii.gz
        ColorPrint "\nCONVERSION OF T1 FILES"
        mrconvert ${1} t1_s.nii.gz -datatype int16 ${arg_stride} -quiet

    else
        mrconvert ${1} dwi_s.nii.gz ${arg_stride} -quiet
        cp $2 encoding_s.b >>${logdir}/input2nii.txt 2>>${logdir}/input2nii.txt
        mrconvert ${3} t1_s.nii.gz ${arg_stride} -quiet\
            >>${logdir}/input2nii.txt 2>>${logdir}/input2nii.txt
    fi

    if [ ${high_res_t1} == 'y' ]
    then
        mv t1_s.nii.gz t1_hr.nii.gz
        scil_resample_volume.py t1_hr.nii.gz t1_s.nii.gz --resolution 1 -f
    fi
}

####################################################################
# Perform either a simple or advanced BET (FSL vs ANTs)
####################################################################
function brain_extraction_tool ()
{
    scil_convert_gradient_mrtrix_to_fsl.py encoding_s.b bval bvec -f \
        >>${logdir}/bvalue_encoding.txt 2>>${logdir}/bvalue_encoding.txt
    scil_extract_b0.py dwi_s.nii.gz bval bvec b0.nii.gz \
        >>${logdir}/extract_b0.txt 2>>${logdir}/extract_b0.txt

    dir_name=$(dirname \
        $(type -p surgery_diffusion_preprocessing.sh))/../data/mni_152_sym_09c/t2
    if [ -d ${dir_name} ] && [ ! $(type -p antsBrainExtraction.sh) == '' ] && [ ! ${1} == 'n' ]
    then
        echo "Template detected, using AntsBrainExtraction instead of FSL"
        antsBrainExtractionTumor.sh -d 3 -a b0.nii.gz -e ${dir_name}/t2_template.nii.gz \
            -m ${dir_name}/t2_brain_probability_map.nii.gz -o bet/ \
            -f ${dir_name}/t2_brain_registration_mask.nii.gz -k 1 \
            >>${logdir}/bet_b0.txt 2>>${logdir}/bet_b0.txt
            mv bet/BrainExtractionPriorWarped.nii.gz b0_bet_mask.nii.gz
            rm -rf bet/
            mrtrix_mult dwi_s.nii.gz b0_bet_mask.nii.gz dwi_bet.nii.gz -quiet
            mrtrix_mult b0.nii.gz b0_bet_mask.nii.gz b0_bet.nii.gz -quiet
    else
        bet_params="b0.nii.gz b0_bet.nii.gz -m -R -f 0.16"

        fsl5.0-bet ${bet_params} >>${logdir}/bet_b0.txt 2>>${logdir}/bet_b0.txt \
            && : || bet ${bet_params} >>${logdir}/bet_b0.txt 2>>${logdir}/bet_b0.txt
        echo "B0 ... Done"
        rm -f dwi_bet.nii.gz
        mrtrix_mult dwi_s.nii.gz b0_bet_mask.nii.gz dwi_bet.nii.gz -quiet
        echo "DWI ... Done"
    fi

    if [ ${1} == 'n' ]
    then
        bet_params="t1_s.nii.gz t1_bet.nii.gz -m -f 0.45 -g 0.1 -R"
        fsl5.0-bet ${bet_params} >>${logdir}/bet_t1.txt 2>>${logdir}/bet_t1.txt \
            && : || bet ${bet_params} >>${logdir}/bet_t1.txt 2>>${logdir}/bet_t1.txt
    else
        dir_name=$(dirname \
            $(type -p surgery_diffusion_preprocessing.sh))/../data/mni_152_sym_09c/t1
        if [ -d ${dir_name} ] && [ ! $(type -p antsBrainExtraction.sh) == '' ]
        then
            echo "Template detected, using AntsBrainExtraction instead of FSL"
            antsBrainExtractionTumor.sh -d 3 -a t1_s.nii.gz -e ${dir_name}/t1_template.nii.gz \
                -m ${dir_name}/t1_brain_probability_map.nii.gz -o bet/ \
                -f ${dir_name}/t1_brain_registration_mask.nii.gz \
                >>${logdir}/bet_t1.txt 2>>${logdir}/bet_t1.txt
                mv bet/BrainExtractionBrain.nii.gz t1_bet.nii.gz
                mv bet/BrainExtractionMask.nii.gz t1_bet_mask.nii.gz
                rm -rf bet/
        else
            echo "Using FSL -B option for robust neck/shoulder cleanup"
            bet_params="t1_s.nii.gz t1_bet.nii.gz -m -f 0.45 -g 0.1 -R -B"
            fsl5.0-bet ${bet_params} >>${logdir}/bet_t1.txt 2>>${logdir}/bet_t1.txt \
                && echo "" || bet ${bet_params} >>${logdir}/bet_t1.txt 2>>${logdir}/bet_t1.txt
        fi
    fi
    echo "T1 ... Done"
}

####################################################################
# Simple data cropping to reduce dimension of volume
####################################################################
function crop_data ()
{
    scil_crop_volume.py dwi_bet.nii.gz dwi_bet_crop.nii.gz \
        --output_bbox dwi_boundingBox.pkl -f >>${logdir}/crop_dwi.txt 2>>${logdir}/crop_dwi.txt
    echo "DWI cropping ... Done"

    scil_crop_volume.py b0_bet.nii.gz b0_bet_crop.nii.gz \
        --output_bbox b0_boundingBox.pkl -f >>${logdir}/crop_b0.txt 2>>${logdir}/crop_b0.txt
    scil_crop_volume.py b0_bet_mask.nii.gz b0_bet_mask_crop.nii.gz \
        --input_bbox b0_boundingBox.pkl -f >>${logdir}/crop_b0.txt 2>>${logdir}/crop_b0.txt
    echo "B0 cropping ... Done"

    scil_crop_volume.py t1_bet.nii.gz t1_bet_crop.nii.gz \
        --output_bbox t1_boundingBox.pkl -f >>${logdir}/crop_t1.txt 2>>${logdir}/crop_t1.txt
    scil_crop_volume.py t1_bet_mask.nii.gz t1_bet_mask_crop.nii.gz \
        --input_bbox t1_boundingBox.pkl -f >>${logdir}/crop_t1.txt 2>>${logdir}/crop_t1.txt
    echo "T1 cropping ... Done"
}

####################################################################
# Denoise the 3D T1 and the 4D DWI, re-extract the B0 once denoised
####################################################################
function denoise_data ()
{
    ColorPrint "\nDIPY DENOISE (~15 min)"
    scil_run_nlmeans.py dwi_bet_crop.nii.gz dwi_bet_crop${was_denoised}.nii.gz \
	    4 --mask b0_bet_mask_crop.nii.gz --noise_est basic \
	    -f >>${logdir}/denoise_dwi.txt 2>>${logdir}/denoise_dwi.txt

    # Better to use the denoised one
    scil_extract_b0.py dwi_bet_crop${was_denoised}.nii.gz bval bvec \
        b0_bet_crop${was_denoised}.nii.gz >>${logdir}/extract_b0.txt \
        2>>${logdir}/extract_b0.txt
    echo "DWIs denoising ... Done"

    scil_run_nlmeans.py t1_bet_crop.nii.gz t1_bet_crop${was_denoised}.nii.gz 1 \
	    --mask t1_bet_mask_crop.nii.gz --noise_est basic \
	    -f >>${logdir}/denoise_t1.txt 2>>${logdir}/denoise_t1.txt
    echo "T1 denoising ... Done"
}

####################################################################
# Upsample DWI to T1 resolution before computation of FODF
####################################################################
function computation_t1_vizu_t1 ()
{
    ColorPrint "\nUPSAMPLE DENOISED DWI TO T1 resolution"
    scil_resample_volume.py dwi_bet_crop${was_denoised}.nii.gz \
        dwi_bet_crop${was_denoised}_t1_res.nii.gz \
        --ref t1_bet_crop${was_denoised}.nii.gz -f \
        >>${logdir}/upsample_dwi.txt 2>>${logdir}/upsample_dwi.txt
    echo "DWIs resample to T1 space ... Done"

    ColorPrint "\nRUN DTI METRICS (~5 min)"
    scil_resample_volume.py b0_bet_mask_crop.nii.gz \
        b0_bet_mask_crop_t1_res.nii.gz --interp nn \
        --ref dwi_bet_crop${was_denoised}_t1_res.nii.gz --enforce_dimension -f \
        >>${logdir}/upsample_b0.txt 2>>${logdir}/upsample_b0.txt
    scil_compute_dti_metrics.py dwi_bet_crop${was_denoised}_t1_res.nii.gz bval bvec \
        --mask b0_bet_mask_crop_t1_res.nii.gz --fa fa_t1_res.nii.gz --rgb rgb_t1_res.nii.gz \
        --not_all -f >>${logdir}/dti_metrics.txt 2>>${logdir}/dti_metrics.txt
    echo "FA, RGB and MD Map computing ... Done"

    ColorPrint "\nCOMPUTE FODFs (~30 min)"
    scil_compute_fodf.py dwi_bet_crop${was_denoised}_t1_res.nii.gz bval bvec \
	    --mask b0_bet_mask_crop_t1_res.nii.gz --frf 15,4,4 \
	    --fodf fodf_t1_res.nii.gz --peaks peaks_t1_res.nii.gz --roi_radius 20 -f \
	    >>${logdir}/fodf_computation.txt 2>>${logdir}/fodf_computation.txt
    echo "FODFs computing ... Done"
}

####################################################################
# Compute FODF in native DWI space
####################################################################
function computation_diff ()
{
    ColorPrint "\nCOMPUTE DTI METRICS (~1 min)"
    scil_compute_dti_metrics.py dwi_bet_crop${was_denoised}.nii.gz bval bvec \
        --mask b0_bet_mask_crop.nii.gz --fa fa_diff_res.nii.gz \
        --rgb rgb_diff_res.nii.gz --md md_diff_res.nii.gz --not_all -f \
        >>${logdir}/dti_metrics.txt 2>>${logdir}/dti_metrics.txt
    echo "FA, RGB, MD Map computing ... Done"

    ColorPrint "\nCOMPUTE FODFs (~5 min)"
    scil_compute_fodf.py dwi_bet_crop${was_denoised}.nii.gz bval bvec \
        --mask b0_bet_mask_crop.nii.gz --fodf fodf_diff_res.nii.gz \
        --peaks peaks_diff_res.nii.gz -f >>${logdir}/fodf_computation.txt \
        2>>${logdir}/fodf_computation.txt
    echo "FODFs computing ... Done"
}

####################################################################
# Resample the FODF to extract peaks, also resample all DWI-related image
####################################################################
function computation_diff_peaks_t1 ()
{
    scil_resample_volume.py fa_diff_res.nii.gz fa_t1_res.nii.gz --ref t1_s.nii.gz \
        -f --enforce_dimension \
        >>${logdir}/resample_fa.txt 2>>${logdir}/resample_fa.txt
    scil_resample_volume.py rgb_diff_res.nii.gz rgb_t1_res.nii.gz --ref t1_s.nii.gz \
        -f \
        >>${logdir}/resample_rgb.txt 2>>${logdir}/resample_rgb.txt
    echo -e "Upsampling RGB and FA map to T1 resolution"
    
    scil_compute_fodf_max_in_ventricles.py fodf_diff_res.nii.gz fa_diff_res.nii.gz \
        md_diff_res.nii.gz  --max_value_output max_in_ventricule.txt -f \
        >>${logdir}/max_ventricule.txt 2>>${logdir}/max_ventricule.txt
    scil_resample_volume.py b0_bet_mask_crop.nii.gz b0_bet_mask_crop_t1_res.nii.gz \
        --ref t1_s.nii.gz --enforce_dimension -f \
        >>${logdir}/resample_b0.txt 2>>${logdir}/resample_b0.txt
    echo -e "Upsampling fodf to T1 resolution"
    scil_resample_volume.py fodf_diff_res.nii.gz fodf_t1_res.nii.gz --ref t1_s.nii.gz \
        -f \
        >>${logdir}/resample_fodf.txt 2>>${logdir}/resample_fodf.txt

    read -d $"\x04" max_in_ventricule < "max_in_ventricule.txt"
    a_threshold=$(bc <<< ${max_in_ventricule}*1.75)
    scil_compute_fodf_metrics.py fodf_t1_res.nii.gz ${a_threshold} \
        --mask b0_bet_mask_crop_t1_res.nii.gz --not_all --peaks peaks_t1_res.nii.gz \
        >>${logdir}/peaks_extraction.txt 2>>${logdir}/peaks_extraction.txt
    echo -e "Peaks extractions ... Done"
}

####################################################################
# Ants registration, $1 is either 'y' or 'n' (SyN vs rigid)
####################################################################
function ants_registration ()
{
    moving=t1_bet_crop${was_denoised}.nii.gz

    if [ $1 == 'y' ]; then
        ColorPrint "\nREGISTRATION OF T1 TO DIFFUSION with nonlinear registration"
        antsRegistration --dimensionality 3 --float 0 \
            --output [output,outputWarped.nii.gz,outputInverseWarped.nii.gz] \
            --interpolation Linear --use-histogram-matching 0 \
            --winsorize-image-intensities [0.005,0.995] \
            --initial-moving-transform [b0_sym.nii.gz,${moving},1] \
            --transform Affine['0.1'] \
            --metric MI[b0_sym.nii.gz,${moving},1,32,Regular,0.25] \
            --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 \
            --smoothing-sigmas 3x2x1x0 \
            --transform Affine['0.1'] \
            --metric MI[b0_sym.nii.gz,${moving},1,32,Regular,0.25] \
            --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 \
            --smoothing-sigmas 3x2x1x0 \
            --transform SyN[0.1,3,0] \
            --metric MI[b0_sym.nii.gz,${moving},1,32] \
            --metric MI[fa_sym.nii.gz,${moving},1,32] \
            --convergence [50x25x10,1e-6,10] --shrink-factors 4x2x1 \
            --smoothing-sigmas 3x2x1 \
            >>${logdir}/linear_registration.txt \
            2>>${logdir}/linear_registration.txt

    elif [ $1 == 'n' ]; then
        ColorPrint "\nREGISTRATION OF T1 TO DIFFUSION with linear registration"
        antsRegistration --dimensionality 3 --float 0 \
            --output [output,outputWarped.nii.gz,outputInverseWarped.nii.gz] \
            --interpolation Linear --use-histogram-matching 0 \
            --winsorize-image-intensities [0.005,0.995] \
            --initial-moving-transform [b0_sym.nii.gz,${moving},1] \
            --transform Rigid['0.1'] \
            --metric MI[b0_sym.nii.gz,${moving},1,32,Regular,0.25] \
            --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 \
            --smoothing-sigmas 3x2x1x0 --transform Affine['0.1'] \
            --metric MI[b0_sym.nii.gz,${moving},1,32,Regular,0.25] \
            --convergence [1000x500x250x100,1e-6,10] --shrink-factors 8x4x2x1 \
            --smoothing-sigmas 3x2x1x0 \
            >>${logdir}/linear_registration.txt \
            2>>${logdir}/linear_registration.txt
    fi

    mv outputWarped.nii.gz t1_bet_crop${was_denoised}${was_warped}.nii.gz
    echo ${2} "registration ... Done"
}

####################################################################
# Segmentation of tissues using FAST (wm, gm, csf)
####################################################################
function tissues_segmentation ()
{
	#WARNING : Always verify the output, especially if a tumor is present
	basename="t1_bet_crop${was_denoised}${was_warped}"
	fast_params="-t 1 -n 3 -H 0.1 -I 4 -l 20.0 -g
	    -o ${basename}.nii.gz ${basename}.nii.gz"
    fsl5.0-fast ${fast_params} >>${logdir}/fast_t1.txt 2>>${logdir}/fast_t1.txt \
        || fast ${fast_params} >>${logdir}/fast_t1.txt 2>>${logdir}/fast_t1.txt

    rm -f mask_*.nii.gz map_*.nii.gz
    basename="t1_bet_crop${was_denoised}${was_warped}"
    mv ${basename}_seg_2.nii.gz mask_wm.nii.gz
    mv ${basename}_seg_1.nii.gz mask_gm.nii.gz
    mv ${basename}_seg_0.nii.gz mask_csf.nii.gz
    mv ${basename}_pve_2.nii.gz map_wm.nii.gz
    mv ${basename}_pve_1.nii.gz map_gm.nii.gz
    mv ${basename}_pve_0.nii.gz map_csf.nii.gz

    # Compute include, exclude and wm/gm interface for PFT tracking
    # Not recommanded for tumor cases, but the command is useful
    #scil_compute_maps_for_particle_filter_tracking.py map_wm.nii.gz map_gm.nii.gz map_csf.nii.gz -f

    echo "Segmentation ... Done"
}

####################################################################
# If a high resolution T1 was used, moved it to diffusion space
# with minimal computation (mainly for MI-Brain)
####################################################################
function process_and_copy_orginal_t1 ()
{
    scil_resample_volume.py t1_bet_mask.nii.gz t1_hr_bet_mask.nii.gz \
        --ref t1_hr.nii.gz --enforce_dimension -f \
        >>${logdir}/bet_hr.txt 2>>${logdir}/bet_hr.txt
    mrtrix_mult t1_hr.nii.gz t1_hr_bet_mask.nii.gz t1_hr_bet.nii.gz -quiet
    scil_resample_volume.py final_data_t1_res/fa_t1_res.nii.gz \
        final_data_t1_res/fa_hr.nii.gz --ref t1_hr.nii.gz --enforce_dimension -f \
        >>${logdir}/fa_hr.txt 2>>${logdir}/fa_hr.txt
    echo -e "BET the high resolution T1 ... Done "

    antsApplyTransforms -d 3 -i t1_hr_bet.nii.gz -r final_data_t1_res/fa_hr.nii.gz \
        -o t1_hr_bet_linear.nii.gz -t output0GenericAffine.mat \
        >>${logdir}/registration_hr.txt 2>>${logdir}/registration_hr.txt
    if [ ${registration} == 'y' ]
    then
        antsApplyTransforms -d 3 -i t1_hr_bet_linear.nii.gz -r \
            final_data_t1_res/fa_hr.nii.gz -o t1_hr_bet_warp.nii.gz -t \
            output1Warp.nii.gz >>${logdir}/registration_hr.txt \
            2>>${logdir}/registration_hr.txt
        cp -f t1_hr_bet_warp.nii.gz final_data_t1_res/t1_hr.nii.gz
    else
        cp -f t1_hr_bet_linear.nii.gz final_data_t1_res/t1_hr.nii.gz
    fi
    rm -f final_data_t1_res/fa_hr.nii.gz
    echo -e "Transform the high resolution T1  ... Done "
}

####################################################################
# Simplify the output by putting main images in a folder together
####################################################################
function copy_to_final_folder ()
{
    # To simplify code, two first cases are identical
    if [ ${upsample1} == 'y' ] || [ ${upsample2} == 'y' ]
    then
        mkdir -p final_data_t1_res

        cp -f t1_bet_crop${was_denoised}${was_warped}.nii.gz final_data_t1_res/t1_s.nii.gz
        cp -f peaks_t1_res.nii.gz fodf_t1_res.nii.gz fa_t1_res.nii.gz \
            rgb_t1_res.nii.gz final_data_t1_res/

        if [ ${segmentation} == 'y' ]
        then
            cp -f mask_*.nii.gz final_data_t1_res/
        fi

        if [ ${high_res_t1} == 'y' ]
        then
            process_and_copy_orginal_t1
        fi
        echo -e "Copying file ... Done"

    else
        # Upsampled only the peaks and DTI metrics
        # so no fodf in the output folder
        if [ ${upsample3} == 'y' ]
        then
            scil_resample_volume.py peaks_diff_res.nii.gz peaks_t1_resNN.nii.gz \
                --interp nn --ref t1_s.nii.gz --enforce_dimension -f >>${logdir}/resample_peaks.txt \
                2>>${logdir}/resample_peaks.txt
            scil_resample_volume.py fa_diff_res.nii.gz fa_t1_res.nii.gz \
                --ref t1_s.nii.gz --enforce_dimension -f >>${logdir}/resample_fa.txt \
                2>>${logdir}/resample_fa.txt
            scil_resample_volume.py rgb_diff_res.nii.gz rgb_t1_res.nii.gz \
                --ref t1_s.nii.gz --enforce_dimension -f >>${logdir}/resample_rgb.txt \
                2>>${logdir}/resample_rgb.txt

            mkdir -p final_data_t1_res

            cp -f t1_bet_crop${was_denoised}${was_warped}.nii.gz final_data_t1_res/t1_s.nii.gz
            cp -f peaks_t1_resNN.nii.gz fa_t1_res.nii.gz \
            rgb_t1_res.nii.gz final_data_t1_res/

            if [ ${segmentation} == 'y' ]
            then
                cp -f mask_*.nii.gz final_data_t1_res/
            fi

            if [ ${high_res_t1} == 'y' ]
            then
                process_and_copy_orginal_t1
            fi
            echo -e "Copying file ... Done"
        else
            # Visualization in DWI resolution, resampling the T1
            scil_resample_volume.py t1_bet_crop${was_denoised}${was_warped}.nii.gz \
                t1_bet_crop${was_denoised}${was_warped}_diff_res.nii.gz --ref fa_diff_res.nii.gz \
                --enforce_dimension -f \
                >>${logdir}/resample_t1.txt 2>>${logdir}/resample_t1.txt
            echo -e "Downsampling T1 to diff_res ... Done "

            mkdir -p final_data_diff_res

            cp -f t1_bet_crop${was_denoised}${was_warped}_diff_res.nii.gz \
                final_data_diff_res/t1_diff_res.nii.gz
            cp -f peaks_diff_res.nii.gz fodf_diff_res.nii.gz fa_diff_res.nii.gz \
                rgb_diff_res.nii.gz final_data_diff_res/

            # Tissues segmentation was performed on the T1
            # Resampling to DWI resolution and renaming
            if [ ${segmentation} == 'y' ]
            then
                scil_resample_volume.py mask_wm.nii.gz mask_wm_diff_res.nii.gz \
                    --ref fa_diff_res.nii.gz --enforce_dimension -f \
                    >>${logdir}/resample_fast.txt 2>>${logdir}/resample_fast.txt
                scil_resample_volume.py mask_gm.nii.gz mask_gm_diff_res.nii.gz \
                    --ref fa_diff_res.nii.gz --enforce_dimension -f \
                    >>${logdir}/resample_fast.txt 2>>${logdir}/resample_fast.txt
                scil_resample_volume.py mask_csf.nii.gz mask_csf_diff_res.nii.gz \
                    --ref fa_diff_res.nii.gz --enforce_dimension -f \
                    >>${logdir}/resample_fast.txt 2>>${logdir}/resample_fast.txt

                cp -f mask_*_diff_res.nii.gz final_data_diff_res/
            fi
            echo -e "Copying file ... Done"
        fi
    fi
}

mrtrix_mult ()
{
    rm -f $3
    if [ ${mrtrix_version} -eq 3 ]; then
        mrcalc $1 $2 -mult $3 -quiet
    else
        mrmult $1 $2 $3
    fi
}

# Useful for 'forward' declaration
main $@
