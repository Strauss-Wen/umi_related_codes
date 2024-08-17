import os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb
import math

def get_data_dict(demos_root: str, obs_len: int, pred_len: int, act_len: int) -> dict:
    print(demos_root)

    if os.path.isfile(demos_root):
        print(f"Load pickeled data from {demos_root} ...")
        data_dict = np.load(demos_root, allow_pickle=True).item()
        return data_dict
    

    demo_types = os.listdir(demos_root)
    data_dict = {}
    count = 0

    for type_i in demo_types:
        type_i_path = os.path.join(demos_root, type_i)
        scenes = os.listdir(type_i_path)
        
        for scene in scenes:
            scene_path = os.path.join(type_i_path, scene)
            cup_T: np.ndarray = np.load(os.path.join(scene_path, "cup_T.npy"))  # [Ni 4 4]
            gripper_T: np.ndarray = np.load((os.path.join(scene_path, "xarm_T.npy")))    # [Ni 4 4]

            # ignore the last row [0 0 0 1] of the transformation matrix
            # cup_T = cup_T[:, :3, :] # [Ni 3 4]
            # gripper_T = gripper_T[:, :3, :]

            inv_cup = np.linalg.inv(cup_T[:-1])
            inv_gripper = np.linalg.inv(gripper_T[:-1])

            # get relative rotations
            cup_T[1:] = cup_T[1:] @ inv_cup
            gripper_T[1:] = gripper_T[1:] @ inv_gripper

            # convert to axis-angle
            cup_rel = R.from_matrix(cup_T[:, :3, :3]).as_euler('xyz')
            gripper_rel = R.from_matrix(gripper_T[:, :3, :3]).as_euler('xyz')

            # compose axis-angle and xyz (3 length + 3 length) 
            act_T = np.concatenate((gripper_rel, np.squeeze(gripper_T[:, :3, 3])), axis=1)

            import pdb; pdb.set_trace()
            # act_T = np.diff(gripper_T, axis=0)  # [Ni-1 3 4]

            scene_length = cup_T.shape[0]
            for i in range(1, scene_length - 1):
                
                obs_pad_start_len = obs_len - i - 1
                if obs_pad_start_len > 0:
                    # need to pad observation at the start

                    obs_i_cup = cup_T[:i+1]   # [i+1 3 4]   
                    obs_i_gripper = gripper_T[:i+1]
                    cup_T_pad = np.repeat(cup_T[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                    gripper_T_pad = np.repeat(gripper_T[0][np.newaxis], repeats=obs_pad_start_len, axis=0)
                    
                    obs_i_cup = np.concatenate([cup_T_pad, obs_i_cup], axis=0)  # [obs_len 3 4]
                    obs_i_gripper = np.concatenate([gripper_T_pad, obs_i_gripper], axis=0)

                else:
                    obs_i_cup = cup_T[(i-obs_len+1):i+1]    # [obs_len 3 4]
                    obs_i_gripper = gripper_T[(i-obs_len+1):i+1]    # [obs_len 3 4]

                obs_i_cup = obs_i_cup.reshape([obs_len, -1])
                obs_i_gripper = obs_i_gripper.reshape(([obs_len, -1]))
                
                act_pad_end_len = (1 + i + pred_len - scene_length)
                if act_pad_end_len > 0:
                    # need to pad action (zeros) at the end

                    act_i = act_T[i:]   # [Ni-1-i 3 4]
                    act_i_pad = np.zeros([act_pad_end_len, 3, 4])
                    act_i = np.concatenate([act_i, act_i_pad], axis=0)  # [pred_len 3 4]

                else:
                    act_i = act_T[i:(pred_len+i)]    # [pred_len 3 4]

                act_i = act_i.reshape([pred_len, -1])


                # set the cup's starting place as origin
                obs_i_cup -= obs_i_cup[0]
                obs_i_gripper -= obs_i_cup[0]
                # pdb.set_trace()

                obs_i = np.concatenate([obs_i_cup, obs_i_gripper], axis=1)   # [obs_len 24]
                
                # obs_i_min = np.min(obs_i, axis=0)
                # obs_i_max = np.max(obs_i, axis=0)
                # obs_i_range = [obs_i_min, obs_i_max]    # [min max]

                # act_i_min = np.min(act_i, axis=0)
                # act_i_max = np.max(act_i, axis=0)
                # act_i_range = [act_i_min, act_i_max]

                # obs_i_normalized = (obs_i - obs_i_range[0]) / (obs_i_range[1] - obs_i_range[0])
                # obs_i_normalized = obs_i_normalized * 2 - 1

                # act_i_normalized = (act_i - act_i_range[0]) / (act_i_range[1] - act_i_range[0])
                # act_i_normalized = act_i_normalized * 2 - 1


                data_dict[count] = {}
                # data_dict[count]['obs'] = obs_i_normalized
                # data_dict[count]['act'] = act_i_normalized
                # data_dict[count]['obs_range'] = obs_i_range
                # data_dict[count]['act_range'] = act_i_range

                data_dict[count]['obs'] = obs_i
                data_dict[count]['act'] = act_i
                
                # if math.isnan(act_i_normalized[0, 0]):
                    # pdb.set_trace()
                    # z=1

                count += 1
    
    return data_dict

if __name__ == "__main__":
    # demos_root = sys.argv[1]
    demos_root = './collected_demos_xarm'
    # demos_root = "./diffusion_data_xarm.npy"

    data_dict = get_data_dict(demos_root=demos_root, obs_len=4, pred_len=16, act_len=8)
    print(len(data_dict.keys()))
    np.save("diffusion_data_xarm.npy", data_dict, allow_pickle=True)
