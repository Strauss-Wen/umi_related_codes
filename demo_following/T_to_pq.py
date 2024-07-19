# convert T [N 4 4] to pose p [N 3] and q [N 4]

import numpy as np
import sapien.core as sapien

def T_to_pq(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # T: [4, 4]
    p = T[:, :3, 3]
    rotation_matrix = T[:, :3, :3]
    qw = np.sqrt(1 + rotation_matrix[:, 0,0] + rotation_matrix[:, 1,1] + rotation_matrix[:, 2,2]) / 2
    qx = (rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2]) / (4 * qw)
    qy = (rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0]) / (4 * qw)
    qz = (rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]) / (4 * qw)
    q = np.array([qw, qx, qy, qz])
    
    return p, q.T

# def T_to_pq_batch(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     # T: [N 4 4]
#     p = T[:, :3, 3]