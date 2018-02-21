 #!/bin/bash
BLUE='\033[0;34m'
LRED='\033[1;31m'
NC='\033[0m'
ColorPrint () { echo -e ${BLUE}${1}${NC}; }
# This script was made to simplify Emmanuel's pipeline
if [ ${#} -ne 4 ]; then
    echo -e "${LRED}Missing some arguments."
    echo -e "  First argument : FA or WM tracking ? ('fa' or 'wm')"
    echo -e "  Second argument : Deterministic or probabilistic tracking ? ('det' or 'prob')'"
    echo -e "  Third argument : Number of streamlines ? (integer)"
    echo -e "  Fourth argument : Number of processes ? (integer)${NC}"
    exit 1
fi

export mrtrix_version
version=$(mrinfo --version | grep 'mrinfo 0.*.*')
[[ ${version} == *.2.* ]] && mrtrix_version=2 || mrtrix_version=3

####################################################################
# Generation of the streamlines
####################################################################
ColorPrint "LAUNCHING ${3} STREAMLINES"
if [ ${1} = "wm" ]
then
    echo "scil_compute_tracking.py fodf_*.nii.gz mask_wm*.nii.gz \
        mask_wm*.nii.gz wm_seed_${2}_track_${3}.trk --algo ${2} --nt ${3} --step 0.5 \
        --minL 20 --maxL 300 --compress 0.2 --processes ${4} -f" > generate_streamlines_log.txt
    scil_compute_tracking.py fodf_*.nii.gz mask_wm*.nii.gz \
        mask_wm*.nii.gz wm_seed_${2}_track_${3}.trk --algo ${2} --nt ${3} --step 0.5 \
        --minL 20 --maxL 300 --compress 0.2 --processes ${4} -f 2>> generate_streamlines_log.txt
else
    rm -f fa_mask_.nii.gz fa_mask.nii.gz
    if [ ${mrtrix_version} -eq 3 ]; then
        mrthreshold fa_*.nii.gz fa_mask_.nii.gz -abs 0.12 -quiet
    else
        threshold fa_*.nii.gz fa_mask_.nii.gz -abs 0.12 -quiet
    fi
	mrconvert fa_mask_.nii.gz fa_mask.nii.gz -datatype uint16 -quiet
	rm -f fa_mask_.nii.gz

    echo "scil_compute_tracking.py fodf_*.nii.gz fa_mask.nii.gz \
        fa_mask.nii.gz fa_seed_${2}_track_${3}.trk --algo ${2} --nt ${3} --step 0.5 \
        --minL 20 --maxL 300 --compress 0.2 --processes ${4} -f" > generate_streamlines_log.txt
    scil_compute_tracking.py fodf_*.nii.gz fa_mask.nii.gz \
        fa_mask.nii.gz fa_seed_${2}_track_${3}.trk --algo ${2} --nt ${3} --step 0.5 \
        --minL 20 --maxL 300 --compress 0.2 --processes ${4} -f 2>> generate_streamlines_log.txt
fi
echo "Streamlines generation ... Done"
rm -f mrtrix-*
