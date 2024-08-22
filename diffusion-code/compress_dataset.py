import numpy as np
import os

demos_root = "/mnt/robotics/ik/out/side-gripper-cam"
seq_list = os.listdir(demos_root)
seq_list.sort()
data_dict = {}
count = 0



for seq in seq_list:
    scene_path = os.path.join(demos_root, seq)
    cup_T: np.ndarray = np.load(os.path.join(scene_path, "box2_T.npy"))  # [Ni 4 4]
    xarm_T: np.ndarray = np.load((os.path.join(scene_path, "xarmsoft_T.npy")))    # [Ni 4 4]
    openness: np.ndarray = np.load(os.path.join(scene_path, "grip_openness.npy"))    # [Ni]

    data_dict[seq] = {}
    data_dict[seq]['box2_T'] = cup_T
    data_dict[seq]['xarmsoft_T'] = xarm_T
    data_dict[seq]['grip_openness'] = openness

np.save("./side-gripper-cam.npy", data_dict, allow_pickle=True)

