# convert the mapping csv file into standard T[camera_ori/camera] matrix
# save in a .npy file [N 4 4] N=frame num
# down sample to 30Hz to equal with realsense

import numpy as np
import pdb
import cv2
import sys

def gopro_marker_as_origin(csv_path: str, ref_T: np.ndarray, first_frame_idx: int = 0):
    '''
    Calculate the T[custom_marker/gopro_pose] for every row in camera_trajectory.csv. \n
    csv_path: path to camera_trajectory.csv \n
    ref_T: [4, 4] T[custom_marker/gopro_initial_pose] calculated by get_reference_T_gopro.py \n
    first_frame_idx: the desired first-frame's index of gopro's record after manual alignment with realsense. \n
    
    Return: [N 4 4] T[custom_marker/gopro_pose] for N poses.
    '''
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    data = data[first_frame_idx::2, -7:-1] # [N (x y z qx qy qz)], down sample to 30Hz
    N = data.shape[0]

    gp_ori_T = np.zeros((N, 4, 4))  # T[camera_ori/camera_current_pos]

    gp_ori_T[:, -1, -1] = 1
    gp_ori_T[:, :3, 3] = data[:, :3]

    for i in range(N):
        gp_ori_T[i, :3, :3], _ = cv2.Rodrigues(data[i, 3:])

    gp_ground_marker_T = np.matmul(ref_T, gp_ori_T)    # T[ground_marker/camera_current_pose]
    return gp_ground_marker_T

if __name__ == "__main__":
    csv_path = sys.argv[1]
    output_path = "./gopro_T.npy"
    # first_frame_idx = 0 # the first aligned frame index with foundationpose
    first_frame_idx = int(sys.argv[2])
    ref_T = np.load(sys.argv[3])

    gp_ground_marker_T = gopro_marker_as_origin(csv_path=csv_path, ref_T=ref_T, first_frame_idx=first_frame_idx)
    np.save(file=output_path, arr=gp_ground_marker_T)
