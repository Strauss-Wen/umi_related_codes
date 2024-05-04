# convert the mapping csv file into standard T[camera_ori/camera] matrix
# save in a .npy file [N 4 4] N=frame num
# down sample to 30Hz to equal with realsense

import numpy as np
import pdb
import cv2
import sys

csv_path = sys.argv[1]
output_path = "./gopro_T.npy"
# first_frame_idx = 0 # the first aligned frame index with foundationpose
first_frame_idx = int(sys.argv[2])
ref_T = np.load(sys.argv[3])
# ref_T = np.array([[ 0.99611698, -0.08436235,  0.02517867,  0.20912424],
#                 [-0.0711143,  -0.93960806, -0.33478269, -0.18282829],
#                 [ 0.05190114,  0.33169216, -0.94195891,  0.37022045],
#                 [ 0.,          0.,          0.,          1.        ]])   # T[ground_marker/camera_ori]

data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
data = data[first_frame_idx::2, -7:-1] # [N (x y z qx qy qz)], down sample to 30Hz
N = data.shape[0]

gp_ori_T = np.zeros((N, 4, 4))  # T[camera_ori/camera_current_pos]

gp_ori_T[:, -1, -1] = 1
gp_ori_T[:, :3, 3] = data[:, :3]

for i in range(N):
    # print(i)
    gp_ori_T[i, :3, :3], _ = cv2.Rodrigues(data[i, 3:])

# pdb.set_trace()    
gp_ground_marker_T = np.matmul(ref_T, gp_ori_T)    # T[ground_marker/camera_current_pose]

np.save(file=output_path, arr=gp_ground_marker_T)
