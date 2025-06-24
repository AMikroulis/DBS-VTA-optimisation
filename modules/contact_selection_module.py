import numpy as npy
import scipy as scy
from scipy import ndimage as nd
from scipy.spatial import Delaunay, ConvexHull
from scipy.io import loadmat
import h5py as hdf
import os
from modules.electrodes import electrode_type
from modules.runtime_storage import storage_space
from modules.woopsies import Woopsies as woops


centroid_replacement_R = npy.empty((8,3))
centroid_replacement_L = npy.empty((8,3))

def electrode_contacts(basepath, subID, side):
    currentpath = os.path.join(basepath,subID)
    reconstruction_path = os.path.join(currentpath,f'reconstruction/{subID}_desc-reconstruction.mat')
    mat = loadmat(reconstruction_path)

    coords = mat['reco'][0]['scrf'][0]['coords_mm'][0][0][0][side]
    
    return coords

def alpha_shape(points, alpha):
    if len(points) < 4:
        return Delaunay(points).simplices

    def add_edge(edges, i, j):
        """ Add an edge between the i-th and j-th points, if not in the list already """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic, id in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        pd = points[id]
        # Compute the circumradius
        a = npy.linalg.norm(pa - pb)
        b = npy.linalg.norm(pb - pc)
        c = npy.linalg.norm(pc - pa)
        d = npy.linalg.norm(pd - pa)
        e = npy.linalg.norm(pd - pb)
        f = npy.linalg.norm(pd - pc)
        s = (a + b + c + d + e + f) / 2.0
        area = npy.sqrt(s * (s - a) * (s - b) * (s - c) * (s - d) * (s - e) * (s - f))
        circum_r = a * b * c * d * e * f / (8.0 * area)
        if circum_r < 1.0 / alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, id)
            add_edge(edges, id, ia)
    return edges

def discretize(original_atlas_coords, alpha=1.0):
    
    # compute the alpha shape
    edges = alpha_shape(original_atlas_coords, alpha)
    # create a mask for the points inside the alpha shape
    mask = npy.zeros(len(original_atlas_coords), dtype=bool)
    for i, j in edges:
        mask[i] = True
        mask[j] = True

    # filter the original coordinates to only include those within the alpha shape
    filtered_coords = original_atlas_coords[mask]

    _offset = npy.min(filtered_coords, axis=0)
    mesh_wo_offset = filtered_coords - _offset
    coords = mesh_wo_offset
    
    x_min, y_min, z_min = npy.min(coords, axis=0)
    x_max, y_max, z_max = npy.max(coords, axis=0)

    xSize = x_max - x_min
    ySize = y_max - y_min
    zSize = z_max - z_min

    grid_size = 0.2
    x_dim = int(npy.ceil(xSize / grid_size))
    y_dim = int(npy.ceil(ySize / grid_size))
    z_dim = int(npy.ceil(zSize / grid_size))

    x = npy.linspace(x_min, x_max, x_dim)
    y = npy.linspace(y_min, y_max, y_dim)
    z = npy.linspace(z_min, z_max, z_dim)
    xv, yv, zv = npy.meshgrid(x, y, z, indexing='ij')
    grid_points = npy.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    delaunay = Delaunay(coords)
    inside_hull = delaunay.find_simplex(grid_points) >= 0

    mesh_3d = inside_hull.reshape((x_dim, y_dim, z_dim))
    mesh_filled = nd.binary_fill_holes(mesh_3d)

    x_scaling = x_dim / xSize
    y_scaling = y_dim / ySize
    z_scaling = z_dim / zSize

    scaling_array = npy.array([x_scaling, y_scaling, z_scaling])

    return mesh_filled, scaling_array, _offset



def load_atlas(_name=None, _path=None, scalefactor = 1, atlas_names=None ,atlas_paths=None):
    if _name is None:
        _name = 'stn'

    atlas_path = atlas_paths[_path]

    name_resolve = dict({'stn': 0, 'motor': 1, 'assoc': 2, 'limbic': 3, 'improv': 0, 'sideeffects': 4})
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

    
    atlas_l, atlas_l_m,  atlas_l_o = discretize(atlas_l_init)
    atlas_r, atlas_r_m,  atlas_r_o = discretize(atlas_r_init)

    return atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o

def centroid(atlas_img, atlas_matrix, atlas_offset):
    discrete_centroid = nd.center_of_mass(atlas_img)

    centroid_coords = discrete_centroid / atlas_matrix + atlas_offset 
    
    return centroid_coords

def full_coords(atlas_img, atlas_matrix, atlas_offset):
    indices = npy.where(atlas_img)
    coords = npy.vstack(indices).T
    coords = coords / atlas_matrix + atlas_offset
    return coords

def calculate_dot_products_in_chunks(improv_vec, sideeffects_vec, chunk_size=512):
    improv_vec = improv_vec.astype(npy.float16)
    sideeffects_vec = sideeffects_vec.astype(npy.float16)

    num_chunks = int(npy.ceil(improv_vec.shape[1] / chunk_size))
    dot_products = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, improv_vec.shape[1])
        improv_chunk = improv_vec[:, start_idx:end_idx, :]

        # calculate dot products for the current chunk
        chunk_dot_products = npy.einsum('ijk,ilk->ijl', improv_chunk, sideeffects_vec)
        dot_products.append(chunk_dot_products)

    # concatenate the results from all chunks along the second axis
    return npy.concatenate(dot_products, axis=1)

def vertex_coords(_name, _path, atlas_paths):
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


    hull_L = scy.spatial.ConvexHull(atlas_l_init)
    hull_R = scy.spatial.ConvexHull(atlas_r_init)

    return hull_L.vertices, hull_R.vertices
    


def contact_angles(el_coords, structure_coords):
    centerpoints = npy.zeros([8,3])
    centerpoints[0] = el_coords[0]
    centerpoints[1] = npy.average(el_coords[1:4], axis=0)
    centerpoints[2] = npy.average(el_coords[1:4], axis=0)
    centerpoints[3] = npy.average(el_coords[1:4], axis=0)
    centerpoints[4] = npy.average(el_coords[4:7], axis=0)
    centerpoints[5] = npy.average(el_coords[4:7], axis=0)
    centerpoints[6] = npy.average(el_coords[4:7], axis=0)
    centerpoints[7] = el_coords[7]

    contact_vector = el_coords - centerpoints

    vector_to_struct = npy.zeros([8,3])
    vector_to_struct = structure_coords - centerpoints

    struct_phi = npy.arctan2(vector_to_struct[:,1], vector_to_struct[:,0]) * 180 / npy.pi
    struct_phi[0] = 90
    struct_phi[7] = 90

    contact_phi = npy.zeros([8])
    contact_phi[1] =npy.arctan2(contact_vector[1,1], contact_vector[1,0]) * 180 / npy.pi
    contact_phi[2] =npy.arctan2(contact_vector[2,1], contact_vector[2,0]) * 180 / npy.pi
    contact_phi[3] =npy.arctan2(contact_vector[3,1], contact_vector[3,0]) * 180 / npy.pi
    contact_phi[4] =npy.arctan2(contact_vector[4,1], contact_vector[4,0]) * 180 / npy.pi
    contact_phi[5] =npy.arctan2(contact_vector[5,1], contact_vector[5,0]) * 180 / npy.pi
    contact_phi[6] =npy.arctan2(contact_vector[6,1], contact_vector[6,0]) * 180 / npy.pi

    unwrapped_rel_phi = struct_phi - contact_phi
    rel_phi = npy.mod(unwrapped_rel_phi+540, 360) -180

    return rel_phi

def calculate_contacts(basepath,subID,atlas_names,atlas_paths, atlas_mode, woopsy, storage_l, storage_r):
    electrode_coords_L = electrode_contacts(basepath,subID,side=1)
    electrode_coords_R = electrode_contacts(basepath,subID,side=0)

    # phi initialisation
    motor_L_phi = 0.
    motor_R_phi = 0.
    assoc_L_phi = 0.
    assoc_R_phi = 0.
    limbic_L_phi = 0.
    limbic_R_phi = 0.
    improv_L_phi = 0.
    improv_R_phi = 0.
    sideeffects_L_phi = 0.
    sideeffects_R_phi = 0.

    if atlas_mode == 'distal':
        atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o = load_atlas('motor','distal_native',1, atlas_names ,atlas_paths)
        motor_coords_L = centroid(atlas_l, atlas_l_m, atlas_l_o)
        motor_L = npy.linalg.norm(electrode_coords_L - motor_coords_L,axis = 1)
        motor_coords_R = centroid(atlas_r, atlas_r_m, atlas_r_o)
        motor_R = npy.linalg.norm(electrode_coords_R - motor_coords_R,axis = 1)
        if electrode_type(basepath,subID).startswith('Abbott'):
            motor_L_phi = contact_angles(electrode_coords_L,motor_coords_L)
            motor_R_phi = contact_angles(electrode_coords_R,motor_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic B'):
            motor_L_phi = contact_angles(electrode_coords_L,motor_coords_L)
            motor_R_phi = contact_angles(electrode_coords_R,motor_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic 3'):
            motor_L_phi = 0.
            motor_R_phi = 0.

        print(f'motor coords L: {motor_coords_L}')
        print(f'electrode coords L:\n {electrode_coords_L}')
        print(f'motor coords R: {motor_coords_R}')
        print(f'electrode coords R:\n {electrode_coords_R}')


        atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o = load_atlas('assoc','distal_native',1, atlas_names ,atlas_paths)
        assoc_coords_L = centroid(atlas_l, atlas_l_m, atlas_l_o)
        assoc_L = npy.linalg.norm(electrode_coords_L - assoc_coords_L,axis = 1)
        assoc_coords_R = centroid(atlas_r, atlas_r_m, atlas_r_o)
        assoc_R = npy.linalg.norm(electrode_coords_R - assoc_coords_R,axis = 1)
        if electrode_type(basepath,subID).startswith('Abbott'):
            assoc_L_phi = contact_angles(electrode_coords_L,assoc_coords_L)
            assoc_R_phi = contact_angles(electrode_coords_R,assoc_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic B'):
            assoc_L_phi = contact_angles(electrode_coords_L,assoc_coords_L)
            assoc_R_phi = contact_angles(electrode_coords_R,assoc_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic 3'):
            assoc_L_phi = 0.
            assoc_R_phi = 0.


        atlas_r, atlas_l, atlas_r_m, atlas_l_m, atlas_r_o, atlas_l_o = load_atlas('limbic','distal_native',1, atlas_names ,atlas_paths)
        limbic_coords_L = centroid(atlas_l, atlas_l_m, atlas_l_o)
        limbic_L = npy.linalg.norm(electrode_coords_L - limbic_coords_L,axis = 1)
        limbic_coords_R = centroid(atlas_r, atlas_r_m, atlas_r_o)
        limbic_R = npy.linalg.norm(electrode_coords_R - limbic_coords_R,axis = 1)
        if electrode_type(basepath,subID).startswith('Abbott'):
            limbic_L_phi = contact_angles(electrode_coords_L,limbic_coords_L)
            limbic_R_phi = contact_angles(electrode_coords_R,limbic_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic B'):
            limbic_L_phi = contact_angles(electrode_coords_L,limbic_coords_L)
            limbic_R_phi = contact_angles(electrode_coords_R,limbic_coords_R)
        if electrode_type(basepath,subID).startswith('Medtronic 3'):
            limbic_L_phi = 0.
            limbic_R_phi = 0.

        return motor_L, motor_R, assoc_L, assoc_R, limbic_L, limbic_R, motor_L_phi, motor_R_phi, assoc_L_phi, assoc_R_phi, limbic_L_phi, limbic_R_phi
    
    

def select_contact_set(score,score_order):
    # keep the 3rd best contact as long as it's closer to the 2nd best than the 2nd  is to the 1st, and provided its rank distance from the best contact does not exceed the best contact's rank.
    difference_0_1 = score[score_order[1]] - score[score_order[0]]
    difference_1_2 = score[score_order[2]] - score[score_order[1]]
    if difference_1_2 <= difference_0_1 and (difference_1_2+difference_0_1 < score[score_order[0]]):
        selected_set = score_order[:3]
    else:
        selected_set = score_order[:2]
    # reduce to 1-2 contacts if the electrode is a Medtronic (non-directional)
    if len(score) == 4:
        if (difference_0_1 < score[score_order[0]]):
            selected_set = score_order[:2]
        else:
            selected_set = npy.array([score_order[0]],dtype='int64')

    return selected_set

def run(_basepath,_subID, _review_weights = [None,None], woopsy = None, atlas_mode = 'distal', storage_l = None, storage_r = None):

    basepath = _basepath
    subID = _subID
    
    if atlas_mode == 'distal':
        atlas_names = dict()
        atlas_names['stn'] = 'STN.nii.gz'
        atlas_names['motor'] = 'STN_motor.nii.gz'
        atlas_names['assoc'] = 'STN_associative.nii.gz'
        atlas_names['limbic'] = 'STN_limbic.nii.gz'

        atlas_paths = dict()
        atlas_paths['distal_native'] = f'{basepath}/{subID}/atlases/DISTAL Minimal (Ewert 2017)/atlas_index.mat'

        mL,mR,aL,aR,lL,lR, mLphi,mRphi,aLphi,aRphi,lLphi,lRphi = calculate_contacts(_basepath,_subID,atlas_names,atlas_paths, atlas_mode, woopsy, storage_l, storage_r)
    
    

    rD = scy.stats.rankdata(npy.abs(mL))
    rPhi = scy.stats.rankdata(npy.abs(mLphi))
    score = rD + rPhi
    print(f'score_pre_LEFT: {score}')
    woopsy.add_info('contact_selection_module',f'initial scores for left side: {score}')
    if _review_weights[1] is None:
        score = score
    else:
        if len(_review_weights[1])== len(score):
            score = score * (_review_weights[1] * 0.5 + 0.5)
            print(f'score_nudged_Left: {score}')
            woopsy.add_info('contact_selection_module',f'nudged scores for left side: {score}')
        else:
            try:
                revised_review_weights = npy.array([_review_weights[1][0], _review_weights[1][1], _review_weights[1][4], _review_weights[1][7]])
                score = score * (revised_review_weights * 0.5 + 0.5)
                print(f'score_nudged_Left: {score}')
                woopsy.add_info('contact_selection_module',f'nudged scores for left side, re-assigned: {score}')
            except:
                score = score
                print(f'Error: {_subID} - left. Review weights must be the same length as the number of contacts')
                woopsy.add_woopsie('contact_selection_module',f'Error in {_subID} - left. Review weights must be the same length as the number of contacts')
    score_order = npy.argsort(score)

    selected_set_L = select_contact_set(score,score_order)

    rD = scy.stats.rankdata(npy.abs(mR))
    rPhi = scy.stats.rankdata(npy.abs(mRphi))
    score = rD + rPhi
    print(f'score_pre_RIGHT: {score}')
    woopsy.add_info('contact_selection_module',f'initial scores for right side: {score}')
    if _review_weights[0] is None:
        score = score
    else:
        if len(_review_weights[0])== len(score):
            score = score * (_review_weights[0] * 0.5 + 0.5)
            print(f'score_nudged_RIGHT: {score}')
            woopsy.add_info('contact_selection_module',f'nudged scores for right side: {score}')
        else:
            try:
                revised_review_weights = npy.array([_review_weights[0][0], _review_weights[0][1], _review_weights[0][4], _review_weights[0][7]])
                score = score * (revised_review_weights * 0.5 + 0.5)
                print(f'score_nudged_RIGHT: {score}')
                woopsy.add_info('contact_selection_module',f'nudged scores for right side, re-assigned: {score}')
            except:
                score = score
                print(f'Error: {_subID} - right. Review weights must be the same length as the number of contacts')
                woopsy.add_woopsie('contact_selection_module',f'Error in {_subID} - right. Review weights must be the same length as the number of contacts')
    score_order = npy.argsort(score)

    selected_set_R = select_contact_set(score,score_order)

    return selected_set_L, selected_set_R


if __name__ == '__main__':
    print('not called')