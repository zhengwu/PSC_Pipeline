#!/usr/bin/env bash
BLUE='\033[0;34m'
LRED='\033[1;31m'
NC='\033[0m'
ColorPrint () { echo -e ${BLUE}${1}${NC}; }
# This script was made to simplify Emmanuel's pipeline
missing=1
if [ ${#} -eq 6 ]; then
    missing=0
fi

if [ ${missing} == 1 ]; then
    echo -e "${LRED}Missing some arguments."
    echo -e "  First argument : Ants linear transformation (*.mat)"
    echo -e "  Second argument : Ants nonlinear transformation (*.nii.gz)"
    echo -e "  Third argument : Tractogram or bundle file (*.trk)"
    echo -e "  Fourth argument : Final reference file after warp (*.nii.gz)"
    echo -e "  Fifth argument : Use the inverse linear transformation ('y' or 'n')?"
    echo -e "  Sixth argument : Output filepath ${NC}"
    exit 1
fi

tracto_name=${3}
tracto_linear_name=${tracto_name/.trk/_linear_temp.trk}
tracto_warp_name=${tracto_name/.trk/_warp_temp.trk}

affine_name=${1/.mat/.npy}
ConvertTransformFile 3 ${1} ${affine_name} --hm --ras

# Registration happen is world space, so application is the right order and
# with the right reference is primordial
ColorPrint "APPLY TRANSFORMATION"
if [ ${5} = "y" ]
then
    echo -e "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1" > identity.txt
    scil_apply_transform_to_tractogram.py ${3} ${2} identity.txt\
        temp.trk -f >> registration_log.txt 2>>registration_log.txt

    scil_apply_warp_to_tractogram.py temp.trk ${4} ${2} \
        ${tracto_warp_name} -f >> registration_log.txt 2>>registration_log.txt
    echo "Nonlinear transformation ... Done"
    rm identity.txt temp.trk

    scil_apply_transform_to_tractogram.py ${tracto_warp_name} ${4} ${affine_name} \
        ${tracto_linear_name} -f >> registration_log.txt 2>>registration_log.txt
    rm ${tracto_warp_name}
    mv ${tracto_linear_name} ${tracto_warp_name}
    echo "Linear transformation ... Done"
else
    scil_apply_transform_to_tractogram.py ${3} ${2} ${affine_name} ${tracto_linear_name} \
        --inverse -f >> registration_log.txt 2>>registration_log.txt
    echo "Linear transformation ... Done"

    scil_apply_warp_to_tractogram.py ${tracto_linear_name} ${4} ${2} \
        ${tracto_warp_name} -f >> registration_log.txt 2>>registration_log.txt
    echo "Nonlinear transformation ... Done"
fi

tracto_warp_ic_name=${6}

# Sometimes points on the edge of a volume can moved outside of the reference frame
scil_remove_invalid_coordinates_from_streamlines.py ${tracto_warp_name} ${4} \
    ${tracto_warp_ic_name}  --gnc --fnc -f >> registration_log.txt 2>>registration_log.txt
echo "Remove invalid coordinates ... Done"

# Clean temporary files
rm ${tracto_warp_name} ${tracto_linear_name} >> registration_log.txt 2>>registration_log.txt
