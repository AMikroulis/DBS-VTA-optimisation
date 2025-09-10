import numpy as npy
import scipy as scy
import pandas as pd
import modules.contact_selection_module as contact_selection_module
from modules.woopsies import Woopsies as woops
import os

def run(basepath, results_path, subid, side, contacts_list, atlas_mode, storage_l, storage_r, woopsy = None):
    print(results_path)
    fieldcsv = pd.read_csv(f'{results_path}/E_field_Lattice.csv')

    array_length = fieldcsv.magnitude.count()
    field_coords = npy.zeros((array_length,3),'float32')

    field_coords[:,0] = fieldcsv['x-pt']
    field_coords[:,1] = fieldcsv['y-pt']
    field_coords[:,2] = fieldcsv['z-pt']

    field_i = npy.zeros([array_length],'float32')
    field_i[:] = fieldcsv['magnitude']

    if atlas_mode == 'distal':
        atlas_paths = dict()
        atlas_paths['distal_native'] = f'{basepath}/{subid}/atlases/DISTAL Minimal (Ewert 2017)/atlas_index.mat'
        
        atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o = contact_selection_module.load_atlas('motor','distal_native',1, None, atlas_paths)


    if side == 0:
        atlas_img_, atlas_matrix_, atlas_offset_ = atlas_r, atlas_r_m, atlas_r_o
    if side == 1:
        atlas_img_, atlas_matrix_, atlas_offset_ = atlas_l, atlas_l_m, atlas_l_o

    # transform field coordinates to atlas voxel space
    field_coords_in_atlas_voxels = npy.rint((field_coords - atlas_offset_) * atlas_matrix_).astype(int)

    # filter coordinates that are within the atlas bounds
    dims = atlas_img_.shape
    valid_indices = npy.all((field_coords_in_atlas_voxels >= 0) & (field_coords_in_atlas_voxels < dims), axis=1)
    
    field_coords_in_atlas_voxels = field_coords_in_atlas_voxels[valid_indices]
    field_magnitudes_in_bounds = field_i[valid_indices]

    # find which of these points are within the target region
    atlas_values_at_field_points = atlas_img_[field_coords_in_atlas_voxels[:, 0], field_coords_in_atlas_voxels[:, 1], field_coords_in_atlas_voxels[:, 2]]
    field_in_target = field_magnitudes_in_bounds[atlas_values_at_field_points > 0]

    if len(field_in_target) > 0:
        median_e_field = npy.median(field_in_target)
        if median_e_field > 1e-6: # avoid division by zero or very small numbers
            current_ratio_estimate = 0.2 / median_e_field
            woopsy.add_info('current_estimation', f'Calculated current ratio for {subid}, side: {side} : {current_ratio_estimate:.2f} -- median field: {median_e_field:.4f} V/mm')
        else:
            current_ratio_estimate = 1.0 # default to 1 if the median field is too low and nag about it in the logs
            woopsy.add_info('current_estimation', f'Calculated current ratio for {subid}, side: {side} : too low, setting to 1 -- median field: {median_e_field:.4f} V/mm (low field)')
            woopsy.add_woopsie('current_estimation', f'Median e-field for {subid}, side: {side} : too low, setting scaling factor to 1 --- check the reconstruction!')
    else:
        # if no overlap between field and atlas, use the default (1) and nag about it in the logs
        current_ratio_estimate = 1.0
        woopsy.add_info('current_estimation', f'Median e-field for {subid}, side: {side} : no overlap, setting to 1')
        woopsy.add_woopsie('current_estimation', f'No overlap between field and atlas for {subid}, side: {side} --- check the reconstruction!')

    return current_ratio_estimate


if __name__ == '__main__':
    print('not called')