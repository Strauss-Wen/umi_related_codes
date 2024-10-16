import numpy as np
import sys
import pdb
import os

path = sys.argv[1]
cup_t = np.load(os.path.join(path, "cup_T.npy"))
gripper_t = np.load(os.path.join(path, "xarm_T.npy"))
data = {}

name = path.split("/")
scene_name = name[-2]+name[-1]
print(scene_name)


data['single_initial'] = {}
data['single_initial']['cup'] = cup_t[0]
data['single_initial']['gripper'] = gripper_t[0]


scene_len = cup_t.shape[0]
middle_idx = int(scene_len/2)
data['single_along_with'] = {}
data['single_along_with']['cup'] = cup_t[middle_idx]
data['single_along_with']['gripper'] = gripper_t[middle_idx]


data['seq_initial'] = {}
data['seq_initial']['cup'] = cup_t[:10]
data['seq_initial']['gripper'] = gripper_t[:10]


data['seq_along_with'] = {}
data['seq_along_with']['cup'] = cup_t[middle_idx-5:middle_idx+5]
data['seq_along_with']['gripper'] = gripper_t[middle_idx-5:middle_idx+5]

np.save(f"./diffusion_eval/{scene_name}.npy", data, allow_pickle=True)