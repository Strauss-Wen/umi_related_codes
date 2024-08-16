# convert T [N 4 4] to pose p [N 3] and q [N 4]

import numpy as np
from scipy.spatial.transform import Rotation as R
import sapien.core as sapien

def T_to_pq(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # T: [4, 4]
    p = T[:, :3, 3]
    rotation_matrix = T[:, :3, :3]
    r = R.from_matrix(rotation_matrix)

    qw = np.sqrt(1 + rotation_matrix[:, 0,0] + rotation_matrix[:, 1,1] + rotation_matrix[:, 2,2]) / 2
    qx = (rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2]) / (4 * qw)
    qy = (rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0]) / (4 * qw)
    qz = (rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]) / (4 * qw)
    q = np.array([qw, qx, qy, qz]).T

    # q = r.as_quat(scalar_first=True) # need to update scipy for scalar_first to work
    
    # optionally scale all p to p - smallest z-coordinate in p
    # min_z = min(p[:,-1])
    # p[:,-1] -= min_z
    
    return p, q