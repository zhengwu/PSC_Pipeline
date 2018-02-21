#fiber tracking
mkdir streamlines
scil_compute_particle_filter_tracking.py --algo 'prob' --npv 8 diffusion/fodf.nii.gz \
   structural/interface.nii.gz structural/map_include.nii.gz structural/map_exclude.nii.gz \
   streamlines/full_interface_prob_pft.trk --processes 4 -f

# remove invalid streamlines
scil_remove_invalid_coordinates_from_streamlines.py --gnc --fnc \
streamlines/full_interface_prob_pft.trk structural/t1_brain.nii.gz streamlines/full_interface_prob_pft_invcoord.trk -f

# remove looped streamlines


# remove unnecessary files
cd streamlines
rm full_interface_prob_pft.trk

cd ..
mkdir connectome
