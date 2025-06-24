import numpy as npy
import scipy as scy
import pandas as pd
import modules.contact_selection_module as contact_selection_module
from modules.woopsies import Woopsies as woops
import os

def run(basepath, results_path, subid, side, contacts_list, atlas_mode, storage_l, storage_r):
    print(results_path)
    fieldcsv = pd.read_csv(f'{results_path}/E_field_Lattice.csv')

    array_length = fieldcsv.magnitude.count()
    field_coords = npy.zeros((array_length,3),'float32')

    field_coords[:,0] = fieldcsv['x-pt']
    field_coords[:,1] = fieldcsv['y-pt']
    field_coords[:,2] = fieldcsv['z-pt']

    field_i = npy.zeros([array_length],'float32')
    field_i[:] = fieldcsv['magnitude']

    vta_0 = npy.argwhere(field_i[:] >=0.2)
    subset_length = len(vta_0)
    vta_field = npy.zeros([subset_length,4],'float32')
    vta_field_coords = npy.zeros([subset_length,3],'float32')

    for _index in range(subset_length):
        old_index = vta_0[_index]
        vta_field[_index,0] = field_coords[old_index,0]
        vta_field[_index,1] = field_coords[old_index,1]
        vta_field[_index,2] = field_coords[old_index,2]
        vta_field[_index,3] = field_i[old_index]
        vta_field_coords[_index,0] = field_coords[old_index,0]
        vta_field_coords[_index,1] = field_coords[old_index,1]
        vta_field_coords[_index,2] = field_coords[old_index,2]

    hull = scy.spatial.ConvexHull(vta_field_coords)
    vertex_points = npy.zeros([len(hull.vertices),3],'float32')

    for point in range(len(hull.vertices)):
        vertex_points[point,0] = vta_field[hull.vertices[point],0]
        vertex_points[point,1] = vta_field[hull.vertices[point],1]
        vertex_points[point,2] = vta_field[hull.vertices[point],2]

    contact_coordinates = []
    for contact in contacts_list:
        contact_coordinates.append(contact_selection_module.electrode_contacts(basepath,subid,side)[contact])

    contact_coordinates = npy.array(contact_coordinates)    
    contacts_midpoint = npy.average(contact_coordinates,axis=0)

    if atlas_mode == 'distal':
        atlas_paths = dict()
        atlas_paths['distal_native'] = f'{basepath}/{subid}/atlases/DISTAL Minimal (Ewert 2017)/atlas_index.mat'
        
        atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o = contact_selection_module.load_atlas('motor','distal_native',1, None, atlas_paths)


    if side == 0:
        atlas_img_, atlas_matrix_, atlas_offset_ = atlas_r, atlas_r_m, atlas_r_o
    if side == 1:
        atlas_img_, atlas_matrix_, atlas_offset_ = atlas_l, atlas_l_m, atlas_l_o

    valid_atlas_points = npy.argwhere(atlas_img_)
    vectors_to_atlas_points = npy.zeros([valid_atlas_points.shape[0],3],'float32')

    for point_i in range(valid_atlas_points.shape[0]):
        vectors_to_atlas_points[point_i] = npy.array([valid_atlas_points[point_i,0],valid_atlas_points[point_i,1],valid_atlas_points[point_i,2]]) / atlas_matrix_ + atlas_offset_

    furthest_point = vectors_to_atlas_points[npy.argmax(npy.linalg.norm(vectors_to_atlas_points,axis=1))]
    vector_to_furthest = furthest_point - contacts_midpoint

    if atlas_mode == 'distal':
        centroid_coords = contact_selection_module.centroid(atlas_img_, atlas_matrix_, atlas_offset_)

    vector_to_centroid = centroid_coords - contacts_midpoint
    
    vector_to_vertices = vertex_points - contacts_midpoint

    target = vector_to_centroid
    target_coords = centroid_coords

    
    vector_angle = npy.zeros(vector_to_vertices.shape[0],'float32')
    for vertex_i in range(vector_to_vertices.shape[0]):
        vector_angle[vertex_i] = npy.arccos(npy.dot(target,vector_to_vertices[vertex_i]) / (npy.linalg.norm(target) * npy.linalg.norm(vector_to_vertices[vertex_i])))

    smallest_angles = npy.argsort(vector_angle)
    closest_vertices = vertex_points[smallest_angles[:3]]
    closest_vertices_midpoint = npy.average(closest_vertices,axis=0)

    distance_to_current_midpoint = npy.linalg.norm(closest_vertices_midpoint - contacts_midpoint)
    distance_to_centroid = npy.linalg.norm(target_coords - contacts_midpoint)
    distance_ratio = distance_to_centroid / distance_to_current_midpoint

    current_ratio_estimate = distance_ratio**2

    return current_ratio_estimate

if __name__ == '__main__':
    print('not called')