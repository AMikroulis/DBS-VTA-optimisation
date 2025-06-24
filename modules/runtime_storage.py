import os
import numpy as npy
import pandas as pd
import pickle
from scipy.spatial import cKDTree
from modules.woopsies import Woopsies as woops

class storage_space:

    def __init__(self, basepath, subid, side):
        self.basepath = basepath
        self.subid = subid
        self.side = side
        self.centroid_replacement = npy.empty((0,3))
        self.icp_matrix = npy.empty((4,4))
        self.kdtree = None
        self.tree_exists = False
        self.icp_exists = False
        self.stored_tree_exists = False
        self.atlas_names = set()
        
    def dispose(self):
        print(f"Disposing resources for {self.subid} on side {self.side}")
        # reset attributes
        self.basepath = None
        self.subid = None
        self.side = None
        self.centroid_replacement = None
        self.icp_matrix = None
        self.kdtree = None
        self.tree_exists = False
        self.icp_exists = False
        self.stored_tree_exists = False
        self.atlas_names = None
        
    def set_centroid_replacement(self, centroid_replacement):
        self.centroid_replacement = centroid_replacement

    def get_centroid_replacement(self):
        return self.centroid_replacement

    def set_icp(self, icp_matrix):
        self.icp_matrix = icp_matrix
        file_path = os.path.join(self.basepath, self.subid, f"icp_matrix_{self.subid}_{self.side}.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.save_icp_to_file(file_path)
        self.icp_exists = True

    def get_icp(self):
        return self.icp_matrix
    
    def has_icp(self):
        if not self.icp_exists:
            file_path = os.path.join(self.basepath, self.subid, f"icp_matrix_{self.subid}_{self.side}.pkl")
            if os.path.exists(file_path):
                self.load_icp_from_file(file_path)
                self.icp_exists = True
        return self.icp_exists


    def save_icp_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.icp_matrix, f)

    def load_icp_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            self.icp_matrix = pickle.load(f)
            self.icp_exists = True

    def save_icp(self):
        if self.icp_exists:
            file_path = os.path.join(self.basepath, self.subid, f"icp_matrix_{self.subid}_{self.side}.pkl")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.save_icp_to_file(file_path)

    def load_icp(self):
        if not self.icp_exists:
            file_path = os.path.join(self.basepath, self.subid, f"icp_matrix_{self.subid}_{self.side}.pkl")
            if os.path.exists(file_path):
                self.load_icp_from_file(file_path)

    def save_tree_to_file(self, tree, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(tree, f)

    def load_tree_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def save_tree(self, tree):
        if self.tree_exists or self.stored_tree_exists:
            pass
        else:
            self.kdtree = tree
            self.tree_exists = True
            file_path = os.path.join(self.basepath, self.subid, f"kdtree_{self.subid}.pkl")
            self.save_tree_to_file(tree, file_path)
            self.stored_tree_exists = True
    
    def load_tree(self):
        if self.kdtree is None:
            file_path = os.path.join(self.basepath, self.subid, f"kdtree_{self.subid}.pkl")
            if os.path.exists(file_path):
                return self.load_tree_from_file(file_path)
        return self.kdtree
    
    def has_tree(self):
        if self.stored_tree_exists and self.kdtree is None:
            file_path = os.path.join(self.basepath, self.subid, f"kdtree_{self.subid}.pkl")
            if os.path.exists(file_path):
                self.kdtree = self.load_tree_from_file(file_path)
                self.tree_exists = True
        file_path = os.path.join(self.basepath, self.subid, f"kdtree_{self.subid}.pkl")
        if os.path.exists(file_path):
            tree = self.load_tree_from_file(file_path)
            self.tree_exists = True

        return self.tree_exists
    
