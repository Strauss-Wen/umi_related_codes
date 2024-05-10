# the original output T from foundationpose can't be directly used with .obj/.dae file generated from blender.
# because all of the T matrices need to be rotated 90 degrees to make the object upwards.

import numpy as np
import sys
import pdb

def T_matrix_adjust(fp_T: np.ndarray) -> np.ndarray:
    rot_T = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]])
    
    rot = fp_T[:, :3, :3]
    rot_new = np.matmul(rot, rot_T)

    fp_T_output = np.zeros_like(fp_T)
    fp_T_output[:, 3, 3] = 1
    fp_T_output[:, :3, 3] = fp_T[:, :3, 3]
    fp_T_output[:, :3, :3] = rot_new

    return fp_T_output

if __name__ == "__main__":
    fp_T_path = sys.argv[1]
    fp_T = np.load(fp_T_path)
    # pdb.set_trace()
    fp_T_output = T_matrix_adjust(fp_T)
    # pdb.set_trace()
    np.save("fp_T_adjust.npy", fp_T_output)
    p = 1