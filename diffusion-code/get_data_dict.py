import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import pdb
from omegaconf import OmegaConf

def get_data_dict(demos_root: str, openness_threshold: float, obs_len: int, pred_len: int, act_len: int) -> dict:
    print(demos_root)
    data_dict_ori = np.load(demos_root, allow_pickle=True).item()

    seq_list = list(data_dict_ori.keys())
    seq_list.sort()
    data_dict = {}
    data_dict['episodes'] = {}
    count = 0

    obs_min = 1000
    obs_max = -1000
    act_min = 1000
    act_max = -1000


    for seq in seq_list:
        # pdb.set_trace()
        obj_T: np.ndarray = data_dict_ori[seq]["box2_T"]        # [Ni 4 4]
        xarm_T: np.ndarray = data_dict_ori[seq]["xarmsoft_T"]   # [Ni 4 4]

        # openness: np.ndarray = np.load(os.path.join(scene_path, "grip_openness.npy"))    # [Ni]

        inv_gripper = np.linalg.inv(xarm_T[:-1])    # [Ni-1 4 4]    T[last_pos/world]

        relative_xarm_pos = copy.deepcopy(xarm_T)
        relative_xarm_pos = xarm_T[1:] @ inv_gripper    # T[world/this_pos] * T[last_pos/world] = T[last_pos/this_pos]
        relative_xarm_pos = np.concatenate((R.from_matrix(relative_xarm_pos[:, :3, :3]).as_euler('xyz'), np.squeeze(relative_xarm_pos[:, :3, 3])), axis=1)  # [Ni-1 6]



        scene_length = obj_T.shape[0]
        for i in range(1, scene_length - 1):
            
            obs_pad_start_len = obs_len - i - 1
            if obs_pad_start_len > 0:
                # need to pad observation at the start

                obs_i_obj = obj_T[:i+1]   # [i+1 6]   
                obs_i_xarm = xarm_T[:i+1]
                obj_T_pad = np.repeat(obj_T[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                xarm_T_pad = np.repeat(xarm_T[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                
                obs_i_obj = np.concatenate([obj_T_pad, obs_i_obj], axis=0)  # [obs_len 6]
                obs_i_xarm = np.concatenate([xarm_T_pad, obs_i_xarm], axis=0)

                # openness_i_obs = openness[:i+1]
                # openness_i_pad = np.repeat(openness[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                # openness_i_obs = np.concatenate([openness_i_pad, openness_i_obs], axis=0)

                
            else:
                obs_i_obj = obj_T[(i-obs_len+1):i+1]        # [obs_len 6]
                obs_i_xarm = xarm_T[(i-obs_len+1):i+1]      # [obs_len 6]
                # openness_i_obs = openness[(i-obs_len+1):i+1]    # [obs_len]



            act_pad_end_len = (1 + i + pred_len - scene_length)
            if act_pad_end_len > 0:
                # need to pad action (zeros) at the end. eular angles of eye(4) is [0 0 0]

                act_i = relative_xarm_pos[i:]   # [Ni-1-i 6]
                act_i_pad = np.zeros([act_pad_end_len, 6])
                act_i = np.concatenate([act_i, act_i_pad], axis=0)  # [pred_len 6]
                
                # openness_i_act = openness[i+1:]
                # openness_i_pad = np.repeat(openness[-1][np.newaxis], repeats=act_pad_end_len, axis=0)
                # openness_i_act = np.concatenate([openness_i_act, openness_i_pad], axis=0)
                # pdb.set_trace()

                # act_i = np.concatenate([act_i, openness_i_act[:, np.newaxis]], axis=1)
                

            else:
                act_i = relative_xarm_pos[i:(pred_len+i)]   # [pred_len 6]

                # openness_i_act = openness[i:(pred_len+i)]   # [pred_len]
                # act_i = np.concatenate([act_i, openness_i_act[:, np.newaxis]], axis=1)    


            # set the gripper's starting place as origin at this episode
            obs_i_anchor = np.linalg.inv(obs_i_xarm[0]) # T[gripper_0/world_origin]

            obs_i_obj = obs_i_obj @ obs_i_anchor    # T[world_origin/obj_now] * T[gripper_0/world_origin] = T[gripper_0/obj_now]
            obs_i_xarm = obs_i_xarm @ obs_i_anchor

            obs_i_obj = np.concatenate((R.from_matrix(obs_i_obj[:, :3, :3]).as_euler('xyz'), np.squeeze(obs_i_obj[:, :3, 3])), axis=1)
            obs_i_xarm = np.concatenate((R.from_matrix(obs_i_xarm[:, :3, :3]).as_euler('xyz'), np.squeeze(obs_i_xarm[:, :3, 3])), axis=1)

            obs_i = np.concatenate([obs_i_obj, obs_i_xarm], axis=1)   # [obs_len 6+6]
            


            data_dict['episodes'][count] = {}
            data_dict['episodes'][count]['obs'] = obs_i
            data_dict['episodes'][count]['act'] = act_i

            # pdb.set_trace()
            obs_min = obs_min if obs_min < np.min(obs_i) else np.min(obs_i)
            obs_max = obs_max if obs_max > np.max(obs_i) else np.max(obs_i)
            act_min = act_min if act_min < np.min(act_i) else np.min(act_i)
            act_max = act_max if act_max > np.max(act_i) else np.max(act_i)

            count += 1


    data_dict['obs_range'] = [obs_min, obs_max]
    data_dict['act_range'] = [act_min, act_max]
    data_dict['obs_len'] = obs_len
    data_dict['pred_len'] = pred_len
    data_dict['act_len'] = act_len


    # normalize observation and action to [-1 1] 
    for i in range(count):
        # pdb.set_trace()
        obs_i = data_dict['episodes'][i]['obs']
        data_dict['episodes'][i]['obs'] = 2 * ((obs_i - obs_min) / (obs_max - obs_min)) - 1   # [-1 ~ 1]
        act_i = data_dict['episodes'][i]['act']
        data_dict['episodes'][i]['act'] = 2 * ((act_i - act_min) / (act_max - act_min)) - 1

    
    return data_dict

if __name__ == "__main__":

    demos_root = "/haozhe-pv/diffusion/side-gripper-cam.npy"
    args = OmegaConf.load("./config.yaml")

    data_dict = get_data_dict(demos_root=demos_root, 
                              openness_threshold=args.openness_threshold, 
                              obs_len=args.lengths.obs_len, 
                              pred_len=args.lengths.pred_len, 
                              act_len=args.lengths.act_len)
    
    print(len(data_dict['episodes'].keys()))

    np.save(f"data_xarm-{args.lengths.obs_len}-{args.lengths.pred_len}-{args.lengths.act_len}.npy", data_dict, allow_pickle=True)

