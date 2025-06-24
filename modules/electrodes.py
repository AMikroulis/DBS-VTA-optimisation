import numpy as npy
from scipy.io import loadmat
import os

def electrode_type(leaddbs_path,ipn_folder):
    reconstruction_path = os.path.join(leaddbs_path,ipn_folder,f'reconstruction/{ipn_folder}_desc-reconstruction.mat')
    mat = loadmat(reconstruction_path)

    electrode_name = mat['reco'][0][0][0][0][0][0][0]

    return electrode_name


def dimensions(leaddbs_path,ipn_folder):
    electrode_name = electrode_type(leaddbs_path,ipn_folder)

    tip_len = dict({})
    contact_len = dict({})
    contact_spacing = dict({})
    diameter = dict({})

    tip_len['Abbott Directed 6172 (short)'] = 1.5
    tip_len['Abbott Directed 6173 (long)'] = 1.5
    tip_len['Medtronic 3387'] = 1.5
    tip_len['Medtronic 3389'] = 1.5
    tip_len['Medtronic 3391'] = 1.5
    tip_len['Medtronic B33005'] = 1.0
    tip_len['Medtronic B33015'] = 1.0

    contact_len['Abbott Directed 6172 (short)'] = 1.5
    contact_len['Abbott Directed 6173 (long)'] = 1.5
    contact_len['Medtronic 3387'] = 1.5
    contact_len['Medtronic 3389'] = 1.5
    contact_len['Medtronic 3391'] = 3.0
    contact_len['Medtronic B33005'] = 1.5
    contact_len['Medtronic B33015'] = 1.5

    contact_spacing['Abbott Directed 6172 (short)'] = 0.5
    contact_spacing['Abbott Directed 6173 (long)'] = 1.5
    contact_spacing['Medtronic 3387'] = 1.5
    contact_spacing['Medtronic 3389'] = 0.5
    contact_spacing['Medtronic 3391'] = 4.0
    contact_spacing['Medtronic B33005'] = 0.5
    contact_spacing['Medtronic B33015'] = 1.5

    diameter['Abbott Directed 6172 (short)'] = 1.29
    diameter['Abbott Directed 6173 (long)'] = 1.29
    diameter['Medtronic 3387'] = 1.27
    diameter['Medtronic 3389'] = 1.27
    diameter['Medtronic 3391'] = 1.27
    diameter['Medtronic B33005'] = 1.36
    diameter['Medtronic B33015'] = 1.36

    return tip_len[electrode_name], contact_len[electrode_name], contact_spacing[electrode_name], diameter[electrode_name]



























if __name__ == "__main__":
    print('not called')