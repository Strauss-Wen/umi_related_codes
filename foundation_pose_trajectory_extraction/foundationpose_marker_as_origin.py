# change the foundationpose's output into given marker as the origin
# save the result in a [N 4 4] array

import os
import pdb
import numpy as np
import sys


def foundationpose_marker_as_origin(data_path: str, ref_T: np.ndarray) -> np.ndarray:
    '''
    Calculate the T[custom_marker/object] for the estimated pose from foundation-pose. \n
    data_path: path to the parent folder of "/ob_in_cam". \n
    ref_T: [4 4] T[ground_marker/camera]. \n
    
    Return: [N 4 4] T[custom_marker/object].
    '''
    files = os.listdir(data_path + "/ob_in_cam")
    files.sort()

    output_T = []

    for file in files:
        file_path = data_path + "/ob_in_cam/" + file

        ori_T = np.loadtxt(file_path)   # T[camera/object]
        relative_T = np.matmul(ref_T, ori_T)   # T[groung_marker/object]
        
        # pdb.set_trace()
        output_T.append(relative_T)

    output_T = np.array(output_T)
    return output_T

if __name__ == "__main__":
    # path to data folder
    # data_path = "/mnt/data/cup_with_clamp"
    data_path = sys.argv[1]
    ref_T_path = sys.argv[2]

    ref_T = np.load(ref_T_path)
    output_T = foundationpose_marker_as_origin(data_path=data_path, ref_T=ref_T)
    np.save(file=data_path+"/fp_ground_marker_origin_T.npy", arr=output_T)



