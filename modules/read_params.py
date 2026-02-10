# -*- coding: utf-8 -*-

import numpy as np 

""" 
The data from the parameter file is read
"""

a= np.loadtxt("./setup.txt", dtype = str)

par_names = 'filters','zero_p','direct','mto_path','mto_path_faezi','ex_visu','ex_profundo','ex_masking','ex_mto','ex_cat',\
            'ex_builder','select_parts','parts_to_build','selection_radii','min_distances','norm_radius','extended_cuts',\
            'ex_builder_extended','dir_builder_extended','width_crop_extended','ex_uni','index_ranges','outer_parts_to_build','outer_part_sample','outer_index_ranges',\
            'use_outer_extended','dir_outer_extended','range_outer_extended','ex_profi','sub_psf',\
            'mag_inf_lim_sub','mag_sup_lim_sub','min_dist_sub','norm_radii_sub','width_image_sub','model_scatter','ex_reflex',\
            'ex_crop_wavelet','use_full_image','size_crop_dec','use_original_image','wavelet_path','use_wiener','ex_galex_pipeline','ex_download_galex','ex_galex_correct','ex_back','use_wiener_bck','ex_sb_mass','use_wavelet','make_wedges','ex_color'
d = dict(zip(par_names,a))

#==============================================================================
# IMAGE/INSTRUMENT INFORMATION
#==============================================================================

filters=str(d['filters'])
zero_p = float(d['zero_p'])
direct=str(d['direct'])
mto_path=str(d['mto_path'])
mto_path_faezi=str(d['mto_path_faezi'])
#==============================================================================
# VISUALIZE IMAGES
#==============================================================================
ex_visu=str(d['ex_visu'])
#==============================================================================
# MEASURE DEPTH
#==============================================================================
ex_profundo= str(d['ex_profundo'])
#==============================================================================
# MASKING 
#==============================================================================
ex_masking=str(d['ex_masking'])
ex_mto=str(d['ex_mto'])
#==============================================================================
# MAKE CATALOGS 
#==============================================================================
ex_cat=str(d['ex_cat'])
#==============================================================================
# BUILDING PSF
#==============================================================================
ex_builder=str(d['ex_builder'])
select_parts=str(d['select_parts'])
parts_to_build=str(d['parts_to_build'])
selection_radii=str(d['selection_radii'])
min_distances=str(d['min_distances'])
norm_radius=str(d['norm_radius'])
extended_cuts=str(d['extended_cuts'])
#==============================================================================
# BUILDING EXTENDED OUTER PART
#==============================================================================
ex_builder_extended=str(d['ex_builder_extended'])
dir_builder_extended=str(d['dir_builder_extended'])
width_crop_extended=str(d['width_crop_extended'])
#==============================================================================
# UNITING PARTS
#==============================================================================
ex_uni=str(d['ex_uni'])
index_ranges=str(d['index_ranges'])
outer_parts_to_build=str(d['outer_parts_to_build'])
outer_part_sample=str(d['outer_part_sample'])
outer_index_ranges=str(d['outer_index_ranges'])
use_outer_extended=str(d['use_outer_extended'])
dir_outer_extended=  str(d['dir_outer_extended'])
range_outer_extended=str(d['range_outer_extended'])
#==============================================================================
#PLOT PROFILES
#==============================================================================
ex_profi=str(d['ex_profi'])

#==============================================================================
# SUBSTRACT STARS
#==============================================================================
sub_psf=str(d['sub_psf'])
mag_inf_lim_sub=str(d['mag_inf_lim_sub'])
mag_sup_lim_sub=str(d['mag_sup_lim_sub'])
min_dist_sub=str(d['min_dist_sub'])
norm_radii_sub=str(d['norm_radii_sub'])
width_image_sub=str(d['width_image_sub'])
model_scatter=str(d['model_scatter'])

#================================
# SUBSTRACT REFLECTIONS
#================================
ex_reflex = str(d['ex_reflex'])
#================================
# Crops and wavelet deconvolution
#================================
ex_crop_wavelet = str(d['ex_crop_wavelet'])
use_full_image = str(d['use_full_image'])
size_crop_dec = str(d['size_crop_dec'])
use_original_image = str(d['use_original_image'])
wavelet_path = str(d['wavelet_path'])
use_wiener=str(d['use_wiener'])

#================================
# DOWNLOAD GALEX DATA
#================================
ex_galex_pipeline=str(d['ex_galex_pipeline'])
ex_download_galex = str(d['ex_download_galex'])
ex_galex_correct = str(d['ex_galex_correct'])

#================================
# ADD ORIGINAL BACKGROUND
#================================
ex_back=str(d['ex_back'])
use_wiener_bck = str(d['use_wiener_bck'])
#================================
# SB AND MASS PROFILES
#================================
ex_sb_mass = str(d['ex_sb_mass'])
use_wavelet=str(d['use_wavelet'])
make_wedges=str(d['make_wedges'])
#================================
# COLOR IMAGES
#================================
ex_color = str(d['ex_color'])