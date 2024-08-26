import numpy as np
import os
from omegaconf import OmegaConf
import sys

log_path = sys.argv[1]
args = OmegaConf.load(f"{log_path}/config.yaml")
scene_id = "1"

data_dict_ori = np.load("/haozhe-pv/diffusion/side-gripper-cam.npy", allow_pickle=True).item()

obj: np.ndarray = data_dict_ori[scene_id]["box2_T"]        # [Ni 4 4]
xarm: np.ndarray = data_dict_ori[scene_id]["xarmsoft_T"]   # [Ni 4 4]
openness: np.ndarray = data_dict_ori[scene_id]["grip_openness"]


scene_len = xarm.shape[0]
middle_idx = scene_len // 2

eval_dict = {}

eval_dict["single_initial"] = {}
eval_dict["seq_initial"] = {}
eval_dict["single_along_with"] = {}
eval_dict["seq_along_with"] = {}

eval_dict["single_initial"]["obj"] = obj
eval_dict["single_initial"]["gripper"] = xarm[0]
eval_dict["single_initial"]["openness"] = openness[0]

eval_dict["single_along_with"]["obj"] = obj[middle_idx:]
eval_dict["single_along_with"]["gripper"] = xarm[middle_idx]
eval_dict["single_along_with"]["openness"] = openness[middle_idx]

eval_dict["seq_initial"]["obj"] = obj
eval_dict["seq_initial"]["gripper"] = xarm[:args.lengths.obs_len]
eval_dict["seq_initial"]["openness"] = openness[:args.lengths.obs_len]

middle_scene_start_idx = middle_idx-int(args.lengths.obs_len / 2)
eval_dict["seq_along_with"]["obj"] = obj[middle_scene_start_idx:]
eval_dict["seq_along_with"]["gripper"] = xarm[middle_scene_start_idx:(middle_scene_start_idx+args.lengths.obs_len)]
eval_dict["seq_along_with"]["openness"] = openness[middle_scene_start_idx:(middle_scene_start_idx+args.lengths.obs_len)]


np.save(f"eval_data-{scene_id}-{args.lengths.obs_len}-{args.lengths.pred_len}-{args.lengths.act_len}.npy", eval_dict, allow_pickle=True)
