import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import pdb

def get_data_dict(demos_root: str, openness_threshold: float, obs_len: int, pred_len: int, act_len: int) -> dict:
    print(demos_root)
    data_dict_ori = np.load(demos_root, allow_pickle=True).item()

    seq_list = list(data_dict_ori.keys())
    seq_list.sort()
    data_dict = {}
    count = 0

    obs_min = 1000
    obs_max = -1000
    act_min = 1000
    act_max = -1000




    for seq in seq_list:

        obj_T: np.ndarray = data_dict_ori[seq]["box2_T"]        # [Ni 4 4]
        xarm_T: np.ndarray = data_dict_ori[seq]["xarmsoft_T"]   # [Ni 4 4]

        # openness: np.ndarray = np.load(os.path.join(scene_path, "grip_openness.npy"))    # [Ni]

        # openness = openness > openness_threshold

        # ignore the last row [0 0 0 1] of the transformation matrix
        obj_T = obj_T[:, :3, :] # [Ni 3 4]
        xarm_T = xarm_T[:, :3, :]

        # save obj and gripper positions
        obj_pos = copy.deepcopy(obj_T)
        obj_pos = np.concatenate((R.from_matrix(obj_pos[:, :3, :3]).as_euler('xyz'), np.squeeze(obj_pos[:, :3, 3])), axis=1)    # [Ni 6]

        gripper_pos = copy.deepcopy(xarm_T)
        gripper_pos = np.concatenate((R.from_matrix(gripper_pos[:, :3, :3]).as_euler('xyz'), np.squeeze(gripper_pos[:, :3, 3])), axis=1)


        # act_T = np.diff(xarm_T, axis=0)  # [Ni-1 3 4]
        gripper_pos_delta = np.diff(gripper_pos, axis=0)    # [Ni-1 6]

        scene_length = obj_pos.shape[0]
        for i in range(1, scene_length - 1):
            
            obs_pad_start_len = obs_len - i - 1
            if obs_pad_start_len > 0:
                # need to pad observation at the start

                obs_i_obj = obj_pos[:i+1]   # [i+1 6]   
                obs_i_xarm = gripper_pos[:i+1]
                obj_T_pad = np.repeat(obj_pos[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                xarm_T_pad = np.repeat(gripper_pos[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                
                obs_i_obj = np.concatenate([obj_T_pad, obs_i_obj], axis=0)  # [obs_len 6]
                obs_i_xarm = np.concatenate([xarm_T_pad, obs_i_xarm], axis=0)

                # openness_i_obs = openness[:i+1]
                # openness_i_pad = np.repeat(openness[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                # openness_i_obs = np.concatenate([openness_i_pad, openness_i_obs], axis=0)

                
            else:
                obs_i_obj = obj_pos[(i-obs_len+1):i+1]        # [obs_len 6]
                obs_i_xarm = gripper_pos[(i-obs_len+1):i+1]      # [obs_len 6]
                # openness_i_obs = openness[(i-obs_len+1):i+1]    # [obs_len]



            act_pad_end_len = (1 + i + pred_len - scene_length)
            if act_pad_end_len > 0:
                # need to pad action (zeros) at the end

                act_i = gripper_pos_delta[i:]   # [Ni-1-i 6]
                act_i_pad = np.zeros([act_pad_end_len, 6])
                act_i = np.concatenate([act_i, act_i_pad], axis=0)  # [pred_len 6]
                
                # openness_i_act = openness[i+1:]
                # openness_i_pad = np.repeat(openness[-1][np.newaxis], repeats=act_pad_end_len, axis=0)
                # openness_i_act = np.concatenate([openness_i_act, openness_i_pad], axis=0)
                # pdb.set_trace()

                # act_i = np.concatenate([act_i, openness_i_act[:, np.newaxis]], axis=1)
                

            else:
                act_i = gripper_pos_delta[i:(pred_len+i)]   # [pred_len 6]
                # openness_i_act = openness[i:(pred_len+i)]   # [pred_len]
                # act_i = np.concatenate([act_i, openness_i_act[:, np.newaxis]], axis=1)    


            # set the obj's starting place as origin at this episode
            obs_i_obj -= obs_i_obj[0]
            obs_i_xarm -= obs_i_obj[0]

            # obs_i = np.concatenate([obs_i_obj, obs_i_xarm, openness_i_obs[:, np.newaxis]], axis=1)   # [obs_len 6+6+1]
            obs_i = np.concatenate([obs_i_obj, obs_i_xarm], axis=1)   # [obs_len 6+6]


            data_dict[count] = {}
            data_dict[count]['obs'] = obs_i
            data_dict[count]['act'] = act_i

            # pdb.set_trace()
            obs_min = obs_min if obs_min < np.min(obs_i) else np.min(obs_i)
            obs_max = obs_max if obs_max > np.max(obs_i) else np.max(obs_i)
            act_min = act_min if act_min < np.min(act_i) else np.min(act_i)
            act_max = act_max if act_max > np.max(act_i) else np.max(act_i)

            # if obs_max > 20:
            #     pdb.set_trace()
            #     z=1
            
            # if act_max > 3:
            #     pdb.set_trace()
            #     z=1



            count += 1


    data_dict['obs_range'] = [obs_min, obs_max]
    data_dict['act_range'] = [act_min, act_max]

    # normalize observation and action to [-1 1] 
    for i in range(count):
        # pdb.set_trace()
        obs_i = data_dict[i]['obs']
        data_dict[i]['obs'] = 2 * ((obs_i - obs_min) / (obs_max - obs_min)) - 1   # [-1 ~ 1]
        act_i = data_dict[i]['act']
        data_dict[i]['act'] = 2 * ((act_i - act_min) / (act_max - act_min)) - 1

    
    return data_dict

if __name__ == "__main__":
    # demos_root = sys.argv[1]
    # demos_root = '/mnt/robotics/ik/out/side-gripper-cam'
    demos_root = "/haozhe-pv/diffusion/side-gripper-cam.npy"

    data_dict = get_data_dict(demos_root=demos_root, openness_threshold=6, obs_len=4, pred_len=8, act_len=8)
    print(len(data_dict.keys()))
    np.save("data_xarm-normalized.npy", data_dict, allow_pickle=True)

