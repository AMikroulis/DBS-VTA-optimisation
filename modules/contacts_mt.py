import numpy as npy
import scipy as scy
from scipy import ndimage as nd
from scipy.spatial import Delaunay, ConvexHull, KDTree
from scipy.ndimage import binary_closing, binary_fill_holes
import nibabel as nib
import json as jsn
import h5py as hdf
import os
from scipy.io import loadmat
import itertools as itt
import pandas as pd
import threading
from joblib import Parallel, delayed
import modules.contact_selection_module as contact_selection_module
import modules.current_estimation as current_estimation
import modules.review_weights as review_weights
from modules.electrodes import dimensions as electrode_dimensions
from modules.electrodes import electrode_type as electrode_type
from modules.runtime_storage import storage_space
from modules.woopsies import Woopsies as woops
import subprocess
import sys

lock = threading.Lock()

ipns = []
contact_pairs_list = []
I_list = []
sidelist = []
motor_ovlps = []
motor_VTA_ovlps = []
assoc_ovlps = []
limbic_ovlps = []

if getattr(sys, 'frozen', False):
    # running as a PyInstaller bundle
    app_path = sys._MEIPASS
else:
    # running with regular Python
    app_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# path to the "resources" folder
resources_path = os.path.join(app_path, 'resources')
sys.path.append(resources_path)

def get_interpreter():
    if getattr(sys, 'frozen', False):  # PyInstaller bundle
        internal_dir = os.path.join(os.path.dirname(sys.executable), '_internal')
        for f in os.listdir(internal_dir):
            if f.startswith('python'):
                return os.path.join(internal_dir, f)
    return sys.executable 

def contact_selection(mh, contactlist, currentslist, electrode_contacts):
    #reset currents
    for contact in range(electrode_contacts):
        mh['Electrodes'][0]['Contacts'][contact]['Current[A]'] = 0.0
        mh['Electrodes'][0]['Contacts'][contact]['Voltage[V]'] = False
    
    mh['Surfaces'][0]['Current[A]'] = 0.0
    
    #set currents
    if len(currentslist) < len(contactlist):    # remove duplicate contacts
        if contactlist[0] == contactlist[4]:
            contactlist.pop(3)
        if contactlist[2] == contactlist[3]:
            contactlist.pop(2)

    for contact in contactlist:
        try:    
            if contact >= 0:
                mh['Electrodes'][0]['Contacts'][contact]['Current[A]'] = currentslist[contactlist.index(contact)]
            if contact == -1:
                mh['Surfaces'][0]['Current[A]'] = currentslist[contactlist.index(contact)]
        except:
            print(f'missing current for contact {contact}')
    if npy.abs(npy.sum(currentslist)) >1e-8:
        print(f'constant current leak: {npy.sum(currentslist) *1000.} mA')

def generate_contact_set_file(basepath, subID, contactslist, currentslist, only_side=None, path_suffix = None, force_monopolar=False):
    currentpath = os.path.join(basepath,subID)
    reconstruction_path = os.path.join(currentpath,f'reconstruction/{subID}_desc-reconstruction.mat')
    mat = loadmat(reconstruction_path)

    def find_ossdbs_folder(basedir, sid):
        stimulations_path = os.path.join(basedir, sid, 'stimulations', 'native')
        if not os.path.exists(stimulations_path):
            return None
        
        for folder in os.listdir(stimulations_path):
            if 'ossdbs' in folder.lower():
                return os.path.join(stimulations_path, folder)
        
        return None

    ossdbs_folder = find_ossdbs_folder(basepath, subID)


    ossdbs_json_path = os.path.join(ossdbs_folder, 'oss-dbs_parameters.json')

    readhdr = jsn.load(open(ossdbs_json_path))
    rh_keys = list(readhdr.keys())

    sides_dict = dict({0:'rh', 1:'lh'})
    filepaths = []
    for side in [0,1]:
        if only_side == 'left' and side == 0:
            continue
        if only_side == 'right' and side == 1:
            continue
        coords = mat['reco'][0]['scrf'][0]['coords_mm'][0][0][0][side]

        electrode_tip_length, electrode_contact_length, electrode_contact_spacing, electrode_diameter  = electrode_dimensions(basepath,subID) 

        mh = readhdr.copy() # copy a header to modify
        [centre_x,centre_y,centre_z] = coords[0]

        mh['BrainRegion']['Center']['x[mm]'] = centre_x
        mh['BrainRegion']['Center']['y[mm]'] = centre_y
        mh['BrainRegion']['Center']['z[mm]'] = centre_z

        # check for medtronic:
        if coords.shape[0] == 4:
            c1_midpoint = coords[1]
        else:
            c1_midpoint = npy.average(coords[1:4], axis=0)

        distance_c1_c0 = npy.linalg.norm(c1_midpoint - coords[0])
        
        scaling_ratio = distance_c1_c0 / 2.0

        tip_length = electrode_tip_length * scaling_ratio
        contact_length = electrode_contact_length * scaling_ratio
        contact_spacing = electrode_contact_spacing * scaling_ratio

        lead_diameter = electrode_diameter * scaling_ratio
        total_length = readhdr[rh_keys[1]][0]['CustomParameters']['total_length'] #(keep from output file)

        # c1_0 to c1 midpoint
        # check for medtronic:
        if coords.shape[0] == 4:
            Rotation_Degrees = 0
            d0_3 = coords[3] - coords[0]
            direction = d0_3 / npy.linalg.norm(d0_3)
        else:
            dc1 = coords[1] - c1_midpoint
            Rotation_Degrees = npy.arctan(dc1[1] / dc1[0]) * 180 / npy.pi + 90
            d0_3 = coords[7] - coords[0]
            direction = d0_3 / npy.linalg.norm(d0_3)

        # tip position
        tip_position = coords[0] - tip_length * direction

        # VTA centre point (average of active contact coords)
        aggregate_coords = []
        for contact in contactslist:
            aggregate_coords.append(coords[contact])
        
        avg_coords = npy.average(aggregate_coords, axis=0)


        # MaterialDistribution
        MRIPath = os.path.join(basepath,subID, 'atlases', 'DISTAL Minimal (Ewert 2017)','segmask_atlas.nii')
        DTIPath = os.path.join(basepath,subID, 'coregistration','dwi', f'{subID}_desc-IITMeanTensor_NormMapping.nii.gz')

        # output direcory
        ossdbs_folder = find_ossdbs_folder(basepath, subID)

        PointModelFileName = os.path.join(ossdbs_folder, 'Allocated_axons.h5')
        OutputPath = os.path.join(ossdbs_folder, f'Results_{sides_dict[side]}')

        mh['Electrodes'][0]['CustomParameters']['tip_length'] = tip_length
        mh['Electrodes'][0]['CustomParameters']['contact_length'] = contact_length
        mh['Electrodes'][0]['CustomParameters']['contact_spacing'] = contact_spacing
        mh['Electrodes'][0]['CustomParameters']['lead_diameter'] = lead_diameter
        mh['Electrodes'][0]['CustomParameters']['total_length'] = total_length
        mh['Electrodes'][0]['Rotation_Degrees'] = Rotation_Degrees

        mh['Electrodes'][0]['Direction']['x[mm]'] = direction[0]
        mh['Electrodes'][0]['Direction']['y[mm]'] = direction[1]
        mh['Electrodes'][0]['Direction']['z[mm]'] = direction[2]

        mh['Electrodes'][0]['TipPosition']['x[mm]'] = tip_position[0]
        mh['Electrodes'][0]['TipPosition']['y[mm]'] = tip_position[1]
        mh['Electrodes'][0]['TipPosition']['z[mm]'] = tip_position[2]

        mh['Solver']['MaximumSteps'] = 1500

        mh['PointModel']['Lattice']['Shape']['x'] = 119
        mh['PointModel']['Lattice']['Shape']['y'] = 119
        mh['PointModel']['Lattice']['Shape']['z'] = 119

        mh['PointModel']['Lattice']['Center']['x[mm]'] = avg_coords[0]
        mh['PointModel']['Lattice']['Center']['y[mm]'] = avg_coords[1]
        mh['PointModel']['Lattice']['Center']['z[mm]'] = avg_coords[2]

        mh['PointModel']['Lattice']['PointDistance[mm]'] = 0.2

        mh['MaterialDistribution']['MRIPath'] = MRIPath
        mh['MaterialDistribution']['DTIPath'] = DTIPath

        mh['PointModel']['Pathway']['Active'] = False
        mh['PointModel']['Pathway']['FileName'] = PointModelFileName

        npy_currents = npy.array(currentslist)
        total_current = npy.round(max(npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents<0))])), npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents>0))]))),5)

        contacts_str = ''
        for contact_ in contactslist:
            contacts_str = contacts_str + str(contact_)
        if path_suffix is not None:
            mh['OutputPath'] = OutputPath+path_suffix+'_'+str(total_current)
        else:
            mh['OutputPath'] = OutputPath+'_c'+contacts_str +'_'+str(total_current)
        
        electrode_name = electrode_type(basepath,subID)
        if electrode_name.startswith('Abbott'):
            n_contacts = 8
        if electrode_name.startswith('Medtronic B'):
            n_contacts = 8
        if electrode_name.startswith('Medtronic 3'):
            n_contacts = 4
        contact_selection(mh,contactlist=contactslist,currentslist=currentslist, electrode_contacts = n_contacts)

        def find_ossdbs_folder(basedir, sid):
            stimulations_path = os.path.join(basedir, sid, 'stimulations', 'native')
            if not os.path.exists(stimulations_path):
                return None
            
            for folder in os.listdir(stimulations_path):
                if 'ossdbs' in folder.lower():
                    return os.path.join(stimulations_path, folder)
            
            return None

        ossdbs_folder = find_ossdbs_folder(basepath, subID)

        mh_file = open(os.path.join(ossdbs_folder, f'ossdbs_{sides_dict[side]}_{str(contactslist)[1:-1]}_{total_current*1000} mA.json'), 'wt')
        jsn.dump(mh, mh_file, indent=4)

        mh_file.close()

        filepaths.append(os.path.join(ossdbs_folder, f'ossdbs_{sides_dict[side]}_{str(contactslist)[1:-1]}_{total_current*1000} mA.json'))

    return filepaths


def discretize(original_atlas_coords, grid_size=0.2):
    # SDF method for oversized atlases
    if len(original_atlas_coords) >50000:
        print('Oversized atlas. Using SDF method.')
        # calculate the bounding box of the input points
        min_coords = npy.min(original_atlas_coords, axis=0)
        max_coords = npy.max(original_atlas_coords, axis=0)
        
        print('constructing grid')
        # create a 3D grid covering the bounding box
        x_grid = npy.arange(min_coords[0], max_coords[0], grid_size)
        y_grid = npy.arange(min_coords[1], max_coords[1], grid_size)
        z_grid = npy.arange(min_coords[2], max_coords[2], grid_size)
        xv, yv, zv = npy.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        grid_points = npy.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

        # KDTree for efficient nearest neighbor search
        print('building tree')
        
        kdtree = KDTree(original_atlas_coords)

        # signed distance for each grid point
        print('computing sdf')

        distances, _ = kdtree.query(grid_points)
        sdf = distances.reshape(xv.shape)

        # threshold the SDF to create a boolean array
        inside_shape = sdf < grid_size
        filled_array = nd.binary_closing(inside_shape[:,:,:], structure=npy.ones((5,5,5))).astype(bool)
        # calculate scaling
        dimensions = npy.array([len(x_grid), len(y_grid), len(z_grid)])
        scaling_array = dimensions / (max_coords - min_coords)
        
        # calculate offset
        _offset = min_coords

    # old version with small meshes (low-res STN atlases)
    if len(original_atlas_coords) <=50000:
        print('Using convex-hull method for low-res atlas')
        xSize = npy.max(original_atlas_coords[0]) - npy.min(original_atlas_coords[0])
        ySize = npy.max(original_atlas_coords[1]) - npy.min(original_atlas_coords[1])
        zSize = npy.max(original_atlas_coords[2]) - npy.min(original_atlas_coords[2])
        _offset = npy.min(original_atlas_coords, axis=0)
        mesh_wo_offset = original_atlas_coords - _offset
        coords = mesh_wo_offset
        hull = scy.spatial.ConvexHull(coords)
        delaunay = scy.spatial.Delaunay(coords[hull.vertices])
        
        x_min, y_min, z_min = npy.min(coords, axis=0)
        x_max, y_max, z_max = npy.max(coords, axis=0)
        xSize = x_max - x_min
        ySize = y_max - y_min
        zSize = z_max - z_min
        max_range = max(xSize, ySize, zSize)
        x_dim = int(npy.ceil(xSize / grid_size))
        y_dim = int(npy.ceil(ySize / grid_size))
        z_dim = int(npy.ceil(zSize / grid_size))
        scaling_factor = max(x_dim / xSize, y_dim / ySize, z_dim / zSize)
        x_dim = int(npy.ceil(xSize * scaling_factor))
        y_dim = int(npy.ceil(ySize * scaling_factor))
        z_dim = int(npy.ceil(zSize * scaling_factor))
        x = npy.linspace(x_min, x_max, x_dim)
        y = npy.linspace(y_min, y_max, y_dim)
        z = npy.linspace(z_min, z_max, z_dim)
        xv, yv, zv = npy.meshgrid(x, y, z, indexing='ij')
        grid_points = npy.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
        inside_hull = delaunay.find_simplex(grid_points) >= 0
        mesh_3d = inside_hull.reshape((x_dim, y_dim, z_dim))
        filled_array = nd.binary_fill_holes(mesh_3d)
        x_scaling = x_dim / xSize
        y_scaling = y_dim / ySize
        z_scaling = z_dim / zSize
        scaling_array = npy.array([x_scaling,y_scaling,z_scaling])

    return filled_array, scaling_array, _offset
        

def load_atlas(_name=None, _path=None, scalefactor = 1, atlas_paths = []):
    if _name is None:
        _name = 'stn'

    atlas_path = atlas_paths[_path]

    name_resolve = dict({'stn': 0, 'motor': 1, 'assoc': 2, 'limbic': 3})

    try:
        atlas_file = hdf.File(atlas_path,'r')

        l_ref = atlas_file['atlases']['XYZ'][()][1][name_resolve[_name]]
        l_values = atlas_file[hdf.h5r.get_name(l_ref,atlas_file.id)]['mm']

        r_ref = atlas_file['atlases']['XYZ'][()][0][name_resolve[_name]]
        r_values = atlas_file[hdf.h5r.get_name(r_ref,atlas_file.id)]['mm']

        atlas_l_init = npy.zeros([l_values.shape[1],l_values.shape[0]])
        atlas_l_init[:,0] = l_values[0]
        atlas_l_init[:,1] = l_values[1]
        atlas_l_init[:,2] = l_values[2]
        
        atlas_r_init = npy.zeros([r_values.shape[1],r_values.shape[0]])
        atlas_r_init[:,0] = r_values[0]
        atlas_r_init[:,1] = r_values[1]
        atlas_r_init[:,2] = r_values[2]
    
    except:
        atlas_file = loadmat(atlas_path)
        l_values = atlas_file['atlases']['XYZ'][()][0][0][name_resolve[_name]][1][0][0][2]
        r_values = atlas_file['atlases']['XYZ'][()][0][0][name_resolve[_name]][0][0][0][2]

        atlas_l_init = npy.zeros([l_values.shape[0],l_values.shape[1]])
        atlas_l_init[:,0] = l_values[:,0]
        atlas_l_init[:,1] = l_values[:,1]
        atlas_l_init[:,2] = l_values[:,2]
        
        atlas_r_init = npy.zeros([r_values.shape[0],r_values.shape[1]])
        atlas_r_init[:,0] = r_values[:,0]
        atlas_r_init[:,1] = r_values[:,1]
        atlas_r_init[:,2] = r_values[:,2]

    atlas_l, atlas_l_m,  atlas_l_o = discretize(atlas_l_init, 0.2)
    atlas_r, atlas_r_m,  atlas_r_o = discretize(atlas_r_init, 0.2)

    return atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o, atlas_r_init, atlas_l_init

def apply_affine_transform(points, affine_matrix):
    # check if the affine matrix is 4x4
    if affine_matrix.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4.")
    
    # convert points to homogeneous coordinates
    num_points = points.shape[0]
    homogeneous_points = npy.hstack([points, npy.ones((num_points, 1))])
    
    # affine transformation
    transformed_homogeneous_points = homogeneous_points @ affine_matrix.T
    
    # convert back from homogeneous coordinates
    return transformed_homogeneous_points[:, :3]

def load_vta(vta_path, scalefactor = 10):
    vta_f = None
    vta_f = nib.load(vta_path)
    vta = vta_f.get_fdata()
    vta_h = vta_f.header

    vta_m = scalefactor*npy.array([vta_h['srow_x'][:3],
                                vta_h['srow_y'][:3],
                                vta_h['srow_z'][:3]])

    vta_o = scalefactor*npy.array([vta_h['srow_x'][-1],
                                vta_h['srow_y'][-1],
                                vta_h['srow_z'][-1]])
    affine_matrix = npy.eye(4)
    affine_matrix[:3,0] = vta_h['srow_x'][:3]
    affine_matrix[:3,1] = vta_h['srow_y'][:3]
    affine_matrix[:3,2] = vta_h['srow_z'][:3]
    affine_matrix[:3,3] = [vta_h['srow_x'][3], vta_h['srow_y'][3], vta_h['srow_z'][3]]

    # vta coords:
    vta_d = npy.argwhere(vta > 0)
    real_space = apply_affine_transform(vta_d, affine_matrix)


    return vta, vta_m, vta_o, real_space


def commonspace(atlas, vta, atlas_m, vta_m, atlas_o, vta_o):
    
    voxel_to_mm = atlas_m
    print('commonspace: applying a 10x magnification for volume measurement')
    adjusted_diag = 10 / voxel_to_mm
    adjusted_offset = 10*atlas_o

    atlas_zoomin_pre = npy.rint(nd.zoom(atlas, adjusted_diag)).astype('bool')
    print('commonspace: getting the magnified point coordinates')
    atlas_zoomin_points = npy.argwhere(atlas_zoomin_pre)
    print(f'commonspace: got {len(atlas_zoomin_points)} magnified points\nStarting discretization of magnified atlas')
    atlas_zoomin_post, zoomin_scale, zoomin_offset = discretize(atlas_zoomin_points, 1)
    print('commonspace: finished discretization of magnified atlas\n Starting padding')
    atlas_zoomin = npy.pad(atlas_zoomin_post, ((zoomin_offset[0],0),(zoomin_offset[1],0),(zoomin_offset[2],0)))

    vta_zoomin = npy.rint(nd.zoom(vta, vta_m.diagonal())).astype('bool')

    offsetdiff = npy.rint(vta_o-adjusted_offset)

    vta_padded = vta_zoomin
    atlas_padded = atlas_zoomin

    offx = int(offsetdiff[0])
    offy = int(offsetdiff[1])
    offz = int(offsetdiff[2])
    
    if offx > 0:
        vta_padded = npy.pad(vta_padded, ((offx,0),(0,0),(0,0)), mode='constant',constant_values=0)
    else:
        atlas_padded = npy.pad(atlas_padded, ((-offx,0),(0,0),(0,0)), mode='constant',constant_values=0)
    if offy > 0:
        vta_padded = npy.pad(vta_padded, ((0,0),(offy,0),(0,0)), mode='constant',constant_values=0)
    else:
        atlas_padded = npy.pad(atlas_padded, ((0,0),(-offy,0),(0,0)), mode='constant',constant_values=0)
    if offz > 0:
        vta_padded = npy.pad(vta_padded, ((0,0),(0,0),(offz,0)), mode='constant',constant_values=0)
    else:
        atlas_padded = npy.pad(atlas_padded, ((0,0),(0,0),(-offz,0)), mode='constant',constant_values=0)
    
    return atlas_padded, vta_padded


def matchdims(atlas, vta):
    xdiff = vta.shape[0] - atlas.shape[0]
    ydiff = vta.shape[1] - atlas.shape[1]
    zdiff = vta.shape[2] - atlas.shape[2]
    if xdiff > 0:
        atlas = npy.pad(atlas, ((0,xdiff),(0,0),(0,0)), mode='constant',constant_values=0)
    if xdiff < 0:    
        vta = npy.pad(vta, ((0,-xdiff),(0,0),(0,0)), mode='constant',constant_values=0)
    if ydiff > 0:
        atlas = npy.pad(atlas, ((0,0),(0,ydiff),(0,0)), mode='constant',constant_values=0)
    if ydiff < 0:
        vta = npy.pad(vta, ((0,0),(0,-ydiff),(0,0)), mode='constant',constant_values=0)
    if zdiff > 0:
        atlas = npy.pad(atlas, ((0,0),(0,0),(0,zdiff)), mode='constant',constant_values=0)
    if zdiff < 0:
        vta = npy.pad(vta, ((0,0),(0,0),(0,-zdiff)), mode='constant',constant_values=0)
    return atlas, vta

def single_folder(basepath, subid, side, contacts_suffix, current_suffix, atlas_paths_dict, selected_atlas_path = 'distal_native', woopsy = None, storage_l = None, storage_r = None):
    path = os.path.join(basepath, subid)
    overlaps = []
    overlaps_vta = []
    
    results_sides_list = [f'Results_rh{contacts_suffix}{current_suffix}', f'Results_lh{contacts_suffix}{current_suffix}']

    if selected_atlas_path == 'distal_native':
        region_options = ['motor','assoc','limbic']
    
    for atlas_name in region_options:

        if selected_atlas_path == 'distal_native':
            atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o, vis_r, vis_l = load_atlas(_name=atlas_name, _path = selected_atlas_path ,atlas_paths=atlas_paths_dict)
            npy.save(f'{basepath}/{subid}/stimulations/visualization_{subid}_{atlas_name}_R.npy', vis_r)
            npy.save(f'{basepath}/{subid}/stimulations/visualization_{subid}_{atlas_name}_L.npy', vis_l)
        print(f'single_folder: Processing {side} {atlas_name}, stored variables retrieved')
        print(f'single_folder: atlas_r shape: {atlas_r.shape}, atlas_l shape: {atlas_l.shape}')
        current_path = path
        try:
            if side == 'right':
                atlas = atlas_r
                atlas_m = atlas_r_m
                atlas_o = atlas_r_o
                results_folder = results_sides_list[0]
            else:
                atlas = atlas_l
                atlas_m = atlas_l_m
                atlas_o = atlas_l_o
                results_folder = results_sides_list[1]
            
            def find_ossdbs_folder(basedir, sid):
                stimulations_path = os.path.join(basedir, sid, 'stimulations', 'native')
                if not os.path.exists(stimulations_path):
                    return None
                
                for folder in os.listdir(stimulations_path):
                    if 'ossdbs' in folder.lower():
                        return os.path.join(stimulations_path, folder)
                
                return None
            
            ossdbs_folder = find_ossdbs_folder(basepath, subid)

            vta, vta_m, vta_o, vis_vta = load_vta(os.path.join(ossdbs_folder, f'{results_folder}','VTA_solution_Lattice.nii'))
            npy.save(f'{current_path}/stimulations/visualization_{subid}_{atlas_name}_{side}_VTA.npy', vis_vta)
            print(f'single_folder: moving to same-space coordinates')
            atlas_padded, vta_padded = commonspace(atlas, vta, atlas_m, vta_m, atlas_o, vta_o)
            print(f'single_folder: matching dimensions')
            atlas_padded, vta_padded = matchdims(atlas_padded, vta_padded)

            overlaps.append(npy.sum(atlas_padded * vta_padded) / npy.sum(atlas_padded))
            if npy.sum(vta_padded) > 0:
                overlaps_vta.append(npy.sum(atlas_padded * vta_padded)  / npy.sum(vta_padded))
            else:
                overlaps_vta.append(0)
            npy.savez_compressed(f'{current_path}/stimulations/OSSDBS_{side}_{atlas_name}_{contacts_suffix}_{current_suffix}_VTA.npz', atlas=atlas_padded, vta=vta_padded)
            npy.savez(f'{current_path}/stimulations/OSSDBS_{side}_{atlas_name}_{contacts_suffix}_{current_suffix}_atlas_matrix.npz', atlas_m=atlas_m, atlas_o=atlas_o)
            npy.savez(f'{current_path}/stimulations/OSSDBS_{side}_{atlas_name}_{contacts_suffix}_{current_suffix}_vta_matrix.npz', vta_m=vta_m, vta_o=vta_o)
        except:
            print(f'Could not read {path}, {side}')
            woopsy.add_woopsie('contacts_mt',f'Error in single folder processing at {path}, {side}. Check the file paths.')

    return overlaps, overlaps_vta[0]


def current_step_run(lefts, rights, current_step_L, current_step_R, subid, basepath, force_monopolar=False, atlas_paths=[], atlas_path_selection='distal_native', outputs = [], woopsy=None, runtime_storage_l=None, runtime_storage_r=None):
    print(f'Running current step for {subid} at {basepath}')
    try:
        if len(lefts) > 0:
            currents_list = []
            if len(lefts) == 2 and force_monopolar==False:
                l1 = lefts[0]
                l2 = lefts[1]
                l3 = None
                p = current_step_L
                n = -p
                currents_list = [n,p]
            
            if len(lefts) == 2 and force_monopolar==True:
                l1 = lefts[0]
                l2 = lefts[1]
                l3 = None
                n = -current_step_L / 2
                p = -2*n
                currents_list = [n,n,p]
            
        
            if len(lefts) == 3 and force_monopolar==False:
                l1 = lefts[0]
                l2 = lefts[1]
                l3 = lefts[2]
                p = current_step_L / 2
                n = -p*2
                currents_list = [n,p,p]

            if len(lefts) == 3 and force_monopolar==True:
                l1 = lefts[0]
                l2 = lefts[1]
                l3 = lefts[2]
                n = -current_step_L / 3
                p = -3*n
                currents_list = [n,n,n,p]

            if len(lefts) == 1:
                l1 = lefts[0]
                n = -current_step_L
                p = current_step_L
                currents_list = [n,p]

            print(f'Left currents: {currents_list}')

            npy_currents = npy.array(currents_list)
            total_current = npy.round(max(npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents<0))])), npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents>0))]))),5)

            Left_contacts = lefts.tolist()
            if force_monopolar==True:
                Left_contacts.append(-1)
            contacts_str = ''
            for contact_ in Left_contacts:
                contacts_str += f'{contact_}'
            contacts_sffx = f'_c{contacts_str}'
            fpaths = generate_contact_set_file(basepath, subid, Left_contacts, currents_list,only_side='left',path_suffix=contacts_sffx, force_monopolar = True)
            
            interpreter_path = sys.executable
            script_path = os.path.join(resources_path, 'ossdbs', 'main.py')
            if getattr(sys, 'frozen', False):
                # Bundled: Use the executable with a flag to run the script
                cmd = [interpreter_path, '--run-script', script_path, fpaths[0]]
            else:
                # Development: Run the script directly with Python
                cmd = [interpreter_path, script_path, fpaths[0]]
            subprocess.run(cmd)
            
            ovlps, ovlps_VTA = single_folder(basepath=basepath, subid= subid, side='left', contacts_suffix=contacts_sffx, current_suffix = f'_{total_current}', atlas_paths_dict=atlas_paths, selected_atlas_path=atlas_path_selection, woopsy=woopsy, storage_l = runtime_storage_l, storage_r = runtime_storage_r)
            lock.acquire()
            ipns.append(subid)
            contact_pairs_list.append(str(Left_contacts)[1:-1])
            sidelist.append('left')
            I_list.append(current_step_L)
            motor_ovlps.append(ovlps[0])
            assoc_ovlps.append(ovlps[1])
            if atlas_path_selection == 'distal_native':
                limbic_ovlps.append(ovlps[2])
                third_overlap = ovlps[2]
            motor_VTA_ovlps.append(ovlps_VTA)
            outputs.append([subid, 'left',contacts_str, current_step_L, ovlps[0], ovlps[1], third_overlap, ovlps_VTA])
            lock.release()
    except:
        print(f'{basepath} L side fail')
        woopsy.add_woopsie('contacts_mt', f'Left side of {subid} failed during current step run')
    
    try:
        if len(rights) > 0:
            currents_list = []
            if len(rights) == 2 and force_monopolar==False:
                r1 = rights[0]
                r2 = rights[1]
                r3 = None
                p = current_step_R
                n = -p
                currents_list = [n,p]
            
            if len(rights) == 2 and force_monopolar==True:
                r1 = rights[0]
                r2 = rights[1]
                r3 = None
                n = -current_step_R / 2
                p = -2*n
                currents_list = [n,n,p]

            if len(rights) == 3 and force_monopolar==False:
                r1 = rights[0]
                r2 = rights[1]
                r3 = rights[2]
                p = current_step_R / 2
                n = -p*2
                currents_list = [n,p,p]

            if len(rights) == 3 and force_monopolar==True:
                r1 = rights[0]
                r2 = rights[1]
                r3 = rights[2]
                n = -current_step_R / 3
                p = -n*3
                currents_list = [n,n,n,p]

            if len(rights) == 1:
                r1 = rights[0]
                n = -current_step_R
                p = current_step_R
                currents_list = [n,p]

            print(f'Right currents: {currents_list}')
            npy_currents = npy.array(currents_list)
            total_current = npy.round(max(npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents<0))])), npy.abs(npy.sum(npy_currents[list(npy.where(npy_currents>0))]))),5)

            Right_contacts = rights.tolist()
            if force_monopolar==True:
                Right_contacts.append(-1)
            contacts_str = ''
            for contact_ in Right_contacts:
                contacts_str += f'{contact_}'
            contacts_sffx = f'_c{contacts_str}'
            fpaths = generate_contact_set_file(basepath, subid, Right_contacts, currents_list,only_side='right',path_suffix=contacts_sffx)
            
            interpreter_path = sys.executable
            script_path = os.path.join(resources_path, 'ossdbs', 'main.py')
            if getattr(sys, 'frozen', False):
                # Bundled: Use the executable with a flag to run the script
                cmd = [interpreter_path, '--run-script', script_path, fpaths[0]]
            else:
                # Development: Run the script directly with Python
                cmd = [interpreter_path, script_path, fpaths[0]]
            subprocess.run(cmd)
            
            ovlps, ovlps_VTA = single_folder(basepath=basepath, subid=subid, side='right', contacts_suffix=contacts_sffx, current_suffix = f'_{total_current}', atlas_paths_dict=atlas_paths, selected_atlas_path=atlas_path_selection, woopsy=woopsy, storage_l=runtime_storage_l, storage_r=runtime_storage_r)
            lock.acquire()
            ipns.append(subid)
            contact_pairs_list.append(str(Right_contacts)[1:-1])
            I_list.append(current_step_R)
            sidelist.append('right')
            motor_ovlps.append(ovlps[0])
            assoc_ovlps.append(ovlps[1])
            if atlas_path_selection == 'distal_native':
                limbic_ovlps.append(ovlps[2])
                third_overlap = ovlps[2]
            motor_VTA_ovlps.append(ovlps_VTA)
            outputs.append([subid, 'right',contacts_str, current_step_R, ovlps[0], ovlps[1], third_overlap, ovlps_VTA])
            lock.release()
    except:
        print(f'{basepath} R side fail')
        woopsy.add_woopsie('contacts_mt', f'Right side of {subid} failed during current step run')
    

def external_call(basedir, selected_ipns, contacts_mode, min_current, max_current, initial_current, multiple_estimates, n_contacts, selected_atlas, progress_ref):
    
    woopsy = woops()

    lock = threading.Lock()

    ipns = []
    contact_pairs_list = []
    I_list = []
    sidelist = []
    motor_ovlps = []
    motor_VTA_ovlps = []
    assoc_ovlps = []
    limbic_ovlps = []
    outputs = []

    atlas_paths = dict()
    atlas_names = dict()
    
    monopolar = True

    if contacts_mode == 'bipolar':
        monopolar = False
    
    if contacts_mode == 'monopolar':
        monopolar = True

    if selected_atlas == 'distal_native':
        atlas_selection = 'distal'
  
    bp = basedir
    sid_list = []
    listdir = os.listdir(bp)
    for folder_maybe in listdir:
        if folder_maybe.startswith('sub-'):
            if folder_maybe in selected_ipns:
                sid_list.append(folder_maybe)

    progress_segments = 2*len(sid_list)
    progress_increment = 100 / progress_segments
    current_progress = 0

    missing_clin_review = []

    for sid in sid_list:
        woopsy.set_subID(sid)
        try:

            atlas_names = dict()
            atlas_names['stn'] = 'STN.nii.gz'
            atlas_names['motor'] = 'STN_motor.nii.gz'
            atlas_names['assoc'] = 'STN_associative.nii.gz'
            atlas_names['limbic'] = 'STN_limbic.nii.gz'
            
            atlas_paths = dict()
            atlas_paths['distal_native'] = f'{bp}/{sid}/atlases/DISTAL Minimal (Ewert 2017)/atlas_index.mat'

            try:
                review_xls = f'{bp}/clinical_review.xlsx'
                cont_R, cont_L, curr_R, curr_L = review_weights.calculate_nudge(review_xls, sid, woopsy)
                if npy.isnan(cont_R).all():
                    cont_R = None
                    curr_R = None
                    missing_clin_review.append(f'{bp} : {sid}, R')
                if npy.isnan(cont_L).all():
                    cont_L = None
                    curr_L = None
                    missing_clin_review.append(f'{bp} : {sid}, L')
                contact_nudge = [cont_R, cont_L]
                current_nudge = [curr_R, curr_L]
                
                print(f'Review xls: {review_xls}')

            except:
                contact_nudge = [None, None]
                current_nudge = [None, None]
                missing_clin_review.append(f'{bp} : {sid}, no clinical review')
                print('no clinical review')
                woopsy.add_woopsie('contacts_mt', f'No clinical review for {sid}')

            woopsy.add_info('contacts_mt',f'info : clinical review checked for {sid}')


            def find_ossdbs_folder(basedir, sid):
                stimulations_path = os.path.join(basedir, sid, 'stimulations', 'native')
                if not os.path.exists(stimulations_path):
                    return None
                
                for folder in os.listdir(stimulations_path):
                    if 'ossdbs' in folder.lower():
                        return os.path.join(stimulations_path, folder)
                
                return None

            # check if the stimulation folder exists:
            ossdbs_folder = find_ossdbs_folder(basedir, sid)
            if ossdbs_folder is None:
                woopsy.add_woopsie('contacts_mt',f'No ossdbs stimulation folder found for subject {subID}!')
                woopsy.add_info('contacts_mt',f'No ossdbs stimulation folder found for subject {subID} : check for folder name misspellings')
            

            runtime_storage_l = storage_space(basepath=basedir,subid=sid,side='L')
            runtime_storage_r = storage_space(basepath=basedir,subid=sid,side='R')

            left_contacts_set, right_contacts_set = contact_selection_module.run(bp,sid, contact_nudge, woopsy, atlas_selection, runtime_storage_l, runtime_storage_r)
            
            woopsy.add_info('contacts_mt',f'info : contact selection completed for {sid}')
            lock = threading.Lock()

            init_current = npy.round(initial_current * 0.001, 4)

            t_pre_L = threading.Thread(target=current_step_run, args=(left_contacts_set, [] , init_current, 0.000, sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
            t_pre_R = threading.Thread(target=current_step_run, args=([], right_contacts_set, 0.000, init_current, sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
            
            t_pre_L.start()
            t_pre_R.start()
            
            t_pre_L.join()
            t_pre_R.join()

            woopsy.add_info('contacts_mt',f'info : initial current estimation completed for {sid}')
            print(f'initial current estimation for {sid} completed')

            Left_contacts = left_contacts_set
            while n_contacts < len(Left_contacts):
                Left_contacts = Left_contacts[:-1]
            contacts_str = ''
            for contact_ in Left_contacts:
                contacts_str += f'{contact_}'
            contacts_sffx_L = f'_c{contacts_str}'
            
            Right_contacts = right_contacts_set
            while n_contacts < len(Right_contacts):
                Right_contacts = Right_contacts[:-1]
            contacts_str = ''
            for contact_ in Right_contacts:
                contacts_str += f'{contact_}'
            contacts_sffx_R = f'_c{contacts_str}'
            
            rh_mono_results_string = f'Results_rh{contacts_sffx_R}-1_{init_current}'
            lh_mono_results_string = f'Results_lh{contacts_sffx_L}-1_{init_current}'

            rh_bi_results_string = f'Results_rh{contacts_sffx_R}_{init_current}'
            lh_bi_results_string = f'Results_lh{contacts_sffx_L}_{init_current}'

            if monopolar == False:
                woopsy.add_info('contacts_mt',f'info : running for bipolar stimulation')
                if len(Right_contacts) > 1:
                    rh_string = rh_bi_results_string
                else:
                    rh_string = rh_mono_results_string
                    woopsy.add_info('contacts_mt',f'info : reverting to monopolar stimulation for right hemisphere -- only 1 contact')
                
                if len(Left_contacts) > 1:
                    lh_string = lh_bi_results_string
                else:
                    lh_string = lh_mono_results_string
                    woopsy.add_info('contacts_mt',f'info : reverting to monopolar stimulation for left hemisphere -- only 1 contact')


            if monopolar == True:
                woopsy.add_info('contacts_mt',f'info : running for monopolar stimulation')
                rh_string = rh_mono_results_string
                lh_string = lh_mono_results_string    
                
            
            results_sides_list = [rh_string, lh_string]

            results_path_R = f'{ossdbs_folder}/{results_sides_list[0]}'
            results_path_L = f'{ossdbs_folder}/{results_sides_list[1]}'
            woopsy.add_info('contacts_mt',f'info : results path created for {sid}')
            
            
            print('File paths set.\nStarting current estimation')


            current_ratio_estimate_R = current_estimation.run(bp, results_path_R, sid, 0, right_contacts_set, atlas_selection,runtime_storage_l, runtime_storage_r)
            current_ratio_estimate_L = current_estimation.run(bp, results_path_L, sid, 1, left_contacts_set, atlas_selection, runtime_storage_l, runtime_storage_r)

            current_estimate_L = npy.round(init_current * current_ratio_estimate_L,4)
            current_estimate_R = npy.round(init_current * current_ratio_estimate_R,4)

            woopsy.add_info('contacts_mt',f'info : current estimation completed for {sid}')

            current_progress += progress_increment
            progress_ref.setValue(current_progress)

            ### make sure the current is within limits
            if current_estimate_L > npy.round(max_current * 0.001, 4):
                current_estimate_L = npy.round(max_current * 0.001, 4)
            if current_estimate_R > npy.round(max_current * 0.001, 4):
                current_estimate_R = npy.round(max_current * 0.001, 4)
            if current_estimate_L < npy.round(min_current * 0.001, 4):
                current_estimate_L = npy.round(min_current * 0.001, 4)
            if current_estimate_R < npy.round(min_current * 0.001, 4):
                current_estimate_R = npy.round(min_current * 0.001, 4)

            adjusted_steps_L = current_estimate_L
            adjusted_steps_R = current_estimate_R
            print(f'current_estimate_L: {current_estimate_L}, current_estimate_R: {current_estimate_R} obtained.\nStarting final VTA estimate')
            t1 = threading.Thread(target=current_step_run, args=(left_contacts_set, [] , adjusted_steps_L, 0.000, sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
            t2 = threading.Thread(target=current_step_run, args=([], right_contacts_set, 0.000, adjusted_steps_R, sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
            
            t1.start()
            t2.start()

            t1.join()
            t2.join()



            if multiple_estimates == True:
                
                woopsy.add_info('contacts_mt',f'info : running multiple VTA estimates for {sid}')

                adjusted_steps_L = [0.0008, 
                                    0.0016,
                                    0.0024, 
                                    0.0032
                                    ]

                adjusted_steps_R = [0.0008, 
                                    0.0016,
                                    0.0024, 
                                    0.0032
                                    ]


                t3 = threading.Thread(target=current_step_run, args=(left_contacts_set, right_contacts_set, adjusted_steps_L[0], adjusted_steps_R[0], sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
                t4 = threading.Thread(target=current_step_run, args=(left_contacts_set, right_contacts_set, adjusted_steps_L[1], adjusted_steps_R[1], sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
                t5 = threading.Thread(target=current_step_run, args=(left_contacts_set, right_contacts_set, adjusted_steps_L[2], adjusted_steps_R[2], sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
                t6 = threading.Thread(target=current_step_run, args=(left_contacts_set, right_contacts_set, adjusted_steps_L[3], adjusted_steps_R[3], sid, bp, monopolar, atlas_paths, selected_atlas, outputs, woopsy, runtime_storage_l, runtime_storage_r))
                
                t3.start()
                t4.start()
                t5.start()
                t6.start()
                
                t3.join()
                t4.join()
                t5.join()
                t6.join()
                
                current_progress += progress_increment
                progress_ref.setValue(current_progress)
                woopsy.add_info('contacts_mt',f'info : multiple estimates completed for {sid}')

            if multiple_estimates == False:
                current_progress += progress_increment
                progress_ref.setValue(current_progress)
                woopsy.add_info('contacts_mt',f'info : final VTA estimate completed for {sid}')

            runtime_storage_l.dispose()
            runtime_storage_l = None
            runtime_storage_r.dispose()
            runtime_storage_r = None


        except:
            print(f'{sid} file(?) fail')
            woopsy.add_woopsie('contacts_mt', f'error : {sid} file failed -- check info log for last successful step')
            
        
        outputs_len = len(outputs)
        for i in range(outputs_len):
            ipns.append(outputs[i][0])
            contact_pairs_list.append(outputs[i][2])
            I_list.append(outputs[i][3])
            sidelist.append(outputs[i][1])
            motor_ovlps.append(outputs[i][4])
            assoc_ovlps.append(outputs[i][5])
            limbic_ovlps.append(outputs[i][6])
            motor_VTA_ovlps.append(outputs[i][7])

        # Clean up
        if runtime_storage_l is not None:
            runtime_storage_l.dispose()
            runtime_storage_l = None

        if runtime_storage_r is not None:
            runtime_storage_r.dispose()
            runtime_storage_r = None

    ipns = npy.array(ipns)
    contact_pairs_list = npy.array(contact_pairs_list)
    I_list = npy.array(I_list) * 1000.
    sidelist = npy.array(sidelist)
    motor_ovlps = npy.array(motor_ovlps)
    assoc_ovlps = npy.array(assoc_ovlps)
    limbic_ovlps = npy.array(limbic_ovlps)
    motor_VTA_ovlps = npy.array(motor_VTA_ovlps)

    if atlas_selection == 'distal':
        df_full = pd.DataFrame({'ipn': ipns, 'contact_pairs': contact_pairs_list,'side': sidelist, 'current (mA)':I_list,'motor': motor_ovlps, 'assoc': assoc_ovlps, 'limbic': limbic_ovlps, 'motor_VTA': motor_VTA_ovlps})
        df = df_full.drop_duplicates(ignore_index=True)
        csv_path = os.path.join(bp, 'overlaps.csv')

        # check if csv file already exists
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            new_ipns = df['ipn'].unique()
            existing_df = existing_df[~existing_df['ipn'].isin(new_ipns)]
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # save back to csv
            combined_df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)

    if len(missing_clin_review) > 0:
        with open(os.path.join(bp, 'missing_clinical_review.txt'), 'w') as missing_review_txt:
            for element in missing_clin_review:
                missing_review_txt.write(f'{element}\n')

    total_woopsies = woopsy.all_woopsies()
    if len(total_woopsies) > 0:
        with open(os.path.join(bp, 'woopsies.log'), 'w') as woopsies_log:
            for woopsie in total_woopsies:
                woopsies_log.write(f'{woopsie}\n')
    total_info = woopsy.all_info()
    if len(total_info) > 0:
        with open(os.path.join(bp, 'info.log'), 'w') as info_log:
            for info in total_info:
                info_log.write(f'{info}\n')

    return 0


if __name__ == '__main__':
    print('not called')
