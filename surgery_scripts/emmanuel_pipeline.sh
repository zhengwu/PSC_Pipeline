#!/usr/bin/env bash
LRED='\033[1;31m'
NC='\033[0m'

function main ()
{
    if [ ${#} -ne 5 ]; then
        echo -e "${LRED}Missing some arguments."
        echo -e "  First argument : Nifti file containing the DWI (*.nii, *.nii.gz)"
        echo -e "  Second argument : File containing the encoding scheme (*.b)"
        echo -e "  Third argument : Nifti file containing the T1 (*.nii, *.nii.gz)"
        echo -e "  Fourth argument : Number of desired streamlines (integer)"
        echo -e "  Fifth argument : Number of processes ? (integer)${NC}"
        exit 1
    fi

    export mrtrix_version
    version=$(mrinfo --version | grep 'mrinfo 0.*.*')
    [[ ${version} == *.2.* ]] && mrtrix_version=2 || mrtrix_version=3
    if [ ${mrtrix_version} -eq 3 ]; then
        arg_stride="-stride 1,2,3,4 -force"
    else
        arg_stride=""
    fi

    # Standardize filename convention for all further steps
    mrconvert ${1} dwi_s.nii.gz ${arg_stride} -quiet
    mrconvert ${3} t1_hr.nii.gz ${arg_stride} -quiet

    # Emmanuel is (almost) always using the same encoding.b
    # This is just for the specific case where he needs it
    #flip_encoding_from_stride ${1} ${2}
    #if [ $? -eq 1 ]
    #then
    #    return 1
    #fi
    cp encoding.b encoding_s.b
    scil_resample_volume.py t1_hr.nii.gz t1_s.nii.gz --resolution 1 -f

    # Lauch the preprocessing with the desired level of complexity
    echo "y y y n n y n y" > temp_params.txt
    surgery_diffusion_preprocessing.sh temp_params.txt
    echo
    rm -f temp_params.txt

    # Automatically generate streamlines as desired for Emmanuel
    cd final_data_t1_res/
    surgery_streamlines_generation.sh fa prob ${4} ${5}
    echo
    cd ..

    # New subdir to facilitate organization
    mkdir -p final_data_t1_space/
    cp -f t1_hr_bet.nii.gz final_data_t1_space/

    # For visualisation with the original high resolution T1, the
    # streamlines are moved back to the T1 space
    surgery_streamlines_deformation.sh output0GenericAffine.mat \
        output1Warp.nii.gz final_data_t1_res/fa_seed_prob_track_${4}.trk \
        t1_bet_crop_denoised.nii.gz y final_data_t1_space/fa_seed_prob_track_${4}_warp_ic.trk

    # Bring at least the FA into the T1 space
    antsApplyTransforms -d 3 -i final_data_t1_res/fa_t1_res.nii.gz -t \
        [output0GenericAffine.mat, 1] -o fa_t1_space_.nii.gz -r ${3} \
        >> registration_log.txt 2>>registration_log.txt
    antsApplyTransforms -d 3 -i fa_t1_space_.nii.gz -t output1InverseWarp.nii.gz \
        -o final_data_t1_space/fa_t1_space.nii.gz -r ${3} \
        >> registration_log.txt 2>>registration_log.txt
}


function flip_encoding_from_stride ()
{
    strides=$(mrinfo ${1} | grep strides)
    strides_cut=$(echo ${strides} | cut -d "[" -f2 | cut -d "]" -f1)
    echo ${strides}
    IFS=', ' read -r -a array <<< ${strides_cut}
    if [ "${array[0]}" -eq 1 ] || [ "${array[0]}" -eq -1 ]
        then
        if [ "${array[0]}" -lt 0 ]; then flip=${flip}' x'; fi
        if [ "${array[1]}" -lt 0 ]; then flip=${flip}' y'; fi
        if [ "${array[2]}" -lt 0 ]; then flip=${flip}' z'; fi
        scil_flip_grad.py ${2} encoding_s.b ${flip} --mrtrix -f
    else
        echo "Strides is not in the right order, fix it manually to be 1,2,3,4"
        return 1
    fi
}

main $@