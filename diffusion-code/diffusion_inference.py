import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from noise_pred_net import ConditionalUnet1D
import pdb
import sys
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import os

# diffusion + euler angles


log_path = sys.argv[1]
eval_path = sys.argv[2]


eval_data_args = os.path.basename(eval_path)[10:-4]
print(eval_data_args)

args = OmegaConf.load(f"{log_path}/config.yaml")

obs_len = args.lengths.obs_len # 4
pred_len = args.lengths.pred_len # 8
act_len = args.lengths.act_len # 8
obs_dim = args.dims.obs_dim # 12
action_dim = args.dims.action_dim # 6
num_diffusion_iters = args.diffusion.num_diffusion_iters # 100

device = torch.device('cuda')


ema_noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_len
    )
ema = EMAModel(
    parameters=ema_noise_pred_net.parameters(),
    power=0.75)

state_dict = torch.load(f"{log_path}/final.pth", weights_only=True)
# pdb.set_trace()

ema.load_state_dict(state_dict)
ema.copy_to(ema_noise_pred_net.parameters())
ema_noise_pred_net.to(device=device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule=args.diffusion.beta_schedule,
    clip_sample=args.diffusion.clip_sample,
    prediction_type=args.diffusion.prediction_type
)

data_dict_ori = np.load(args.data.path, allow_pickle=True).item()
obs_range = data_dict_ori['obs_range']
act_range = data_dict_ori['act_range']


def diffusion_inference(obj_gt_traj: np.ndarray, gripper_init_T: np.ndarray, gripper_openness: np.ndarray, max_iter: int):
    # obj_gt_traj: [N 4 4]
    # gripper_init_T: [Ni 4 4]


    B = 1
    # pdb.set_trace()

    # normalize observation
    if (len(gripper_init_T.shape)) == 2:
        # [4 4]
        # obs_obj = np.concatenate((R.from_matrix(obj_gt_traj[:3, :3]).as_euler('xyz'), np.squeeze(obj_gt_traj[:3, 3])), axis=0)    # [6]
        # obs_gripper = np.concatenate((R.from_matrix(gripper_init_T[:3, :3]).as_euler('xyz'), np.squeeze(gripper_init_T[:3, 3])), axis=0)    # [6]


        # obs_obj = np.repeat(obs_obj[np.newaxis], repeats=obs_len, axis=0)   # [obs_len 6]
        obs_gripper = np.repeat(gripper_init_T[np.newaxis], repeats=obs_len, axis=0)   # [obs_len 4 4]

        # openness = np.repeat(gripper_openness[np.newaxis], repeats=obs_len, axis=0) # [obs_len 1]
        # obs = np.concatenate([obs_obj, obs_gripper, openness[:, np.newaxis]], axis=1)  # [obs_len 13]
    
    else:
        # [N 4 4]
        # obs_obj = np.concatenate((R.from_matrix(obj_gt_traj[:, :3, :3]).as_euler('xyz'), np.squeeze(obj_gt_traj[:, :3, 3])), axis=1)    # [N 6]
        # obs_gripper = np.concatenate((R.from_matrix(gripper_init_T[:, :3, :3]).as_euler('xyz'), np.squeeze(gripper_init_T[:, :3, 3])), axis=1)    # [N 6]
        # pdb.set_trace()

        obs_gripper = gripper_init_T

        # openness = gripper_openness

        # obs = np.concatenate([obs_obj, obs_gripper, openness[:, np.newaxis]], axis=1)  # [obs_len 13]
    
    # pdb.set_trace()
    history_gripper_traj = [obs_gripper[i] for i in range(obs_len)] # [obs_len [4 4]]  # position in world fame
    # history_openness = [openness[i] for i in range(obs_len)]    # [obs_len]

    # obj_gt_traj = np.concatenate((R.from_matrix(obj_gt_traj[:, :3, :3]).as_euler('xyz'), np.squeeze(obj_gt_traj[:, :3, 3])), axis=1)    # [N 6]


    frame_idx = obs_len
    with torch.no_grad():
        for iter_num in tqdm(range(max_iter), desc="iter", miniters=1):
            
            # use the true trajectory of the object rather than generate its traj in interactive simulation
            if frame_idx <= obj_gt_traj.shape[0]:
                obs_i_obj = obj_gt_traj[frame_idx-obs_len:frame_idx]
            else:
                obs_i_obj = obj_gt_traj[-obs_len:]

            # pdb.set_trace()

            obs_i_xarm = np.array(history_gripper_traj[-obs_len:])   # [obs_len 4 4]

            # set xarm's starting pose as the anchor
            anchor_pose = np.linalg.inv(obs_i_xarm[0])  # [4 4]
            # pdb.set_trace()

            obs_i_xarm = obs_i_xarm @ anchor_pose     # T[world_frame/xarm_present] * T[xarm_0/world_frame]
            obs_i_obj = obs_i_obj @ anchor_pose     # T[xarm_0/obj_present]

            obs_i_obj = np.concatenate((R.from_matrix(obs_i_obj[:, :3, :3]).as_euler('xyz'), np.squeeze(obs_i_obj[:, :3, 3])), axis=1)
            obs_i_xarm = np.concatenate((R.from_matrix(obs_i_xarm[:, :3, :3]).as_euler('xyz'), np.squeeze(obs_i_xarm[:, :3, 3])), axis=1)

            # obs_i_openness = np.array(history_openness[-obs_len:])      # [obs_len 1]
            # pdb.set_trace()
            
            # obs_i = np.concatenate([obs_i_obj, obs_i_gripper, obs_i_openness[:, np.newaxis]], axis=1)  # [obs_len 13]
            obs_i = np.concatenate([obs_i_obj, obs_i_xarm], axis=1)  # [obs_len 12]
            obs_i = torch.from_numpy(obs_i).to(device=device, dtype=torch.float32)

            # normalize the observation
            obs_i = 2 * ((obs_i - obs_range[0]) / (obs_range[1] - obs_range[0])) - 1


            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_i.unsqueeze(0) # [1 obs_len 13]
            obs_cond = obs_cond.flatten(start_dim=1)

            # initialize action from Guassian noise
            naction_rand = torch.randn((B, pred_len, action_dim), device=device)
            naction = naction_rand

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters, device=device)


            for k in noise_scheduler.timesteps:
                # predict noise
                # pdb.set_trace()
                noise_pred = ema_noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample


            naction = naction[0].detach().to('cpu').numpy()    # [pred_len 6]
            action = naction[:act_len]  # [act_len 6]
            # pdb.set_trace()

            # unnormalize
            action = ((action + 1) / 2) * (act_range[1] - act_range[0]) + act_range[0]  # [act_len 6]
            action_mat = np.zeros([act_len, 4, 4])
            action_mat[:, :3, :3] = R.from_euler('xyz', action[:, :3]).as_matrix()
            action_mat[:, :3, 3] = action[:, 3:]
            action_mat[:, 3, 3] = 1     # [act_len 4 4]



            # execute action_horizon number of steps without replanning
            for i in range(act_len):
                frame_idx += 1
                # pdb.set_trace()
                # gripper_pose_i = history_gripper_traj[-1] + action[i, :-1]
                # gripper_pose_i = history_gripper_traj[-1] @ action_mat[i]
                gripper_pose_i = action_mat[i] @ history_gripper_traj[-1]

                history_gripper_traj.append(gripper_pose_i)

                # history_openness.append(action[i, -1])



    # pad/reshape from [N 12] to [N 4 4]
    # frame_num = len(history_gripper_traj)
    history_gripper_traj = np.array(history_gripper_traj)   # [N 4 4]
    # rot = R.from_euler('xyz', history_gripper_traj[:, :3]).as_matrix()  # [N 3 3]
    # pos = history_gripper_traj[:, 3:]   # [N 3]

    # gripper_traj_mat = np.zeros([frame_num, 4, 4])
    # gripper_traj_mat[:, :3, :3] = rot
    # gripper_traj_mat[:, :3, 3] = pos
    # gripper_traj_mat[:, 3, 3] = 1   # [N 4 4]

    output = {}
    output['gripper_traj'] = history_gripper_traj
    # output['openness'] = gripper_openness


    return output





eval_data = np.load(eval_path, allow_pickle=True).item()

eval_output = {}
max_iter = 20


print("single initial ...")
gripper_traj = diffusion_inference(obj_gt_traj=eval_data['single_initial']['obj'], gripper_init_T=eval_data['single_initial']['gripper'], gripper_openness=eval_data["single_initial"]["openness"], max_iter=max_iter)
eval_output['single_initial'] = gripper_traj

print("sequence initial ...")
gripper_traj = diffusion_inference(obj_gt_traj=eval_data['seq_initial']['obj'], gripper_init_T=eval_data['seq_initial']['gripper'], gripper_openness=eval_data["seq_initial"]["openness"], max_iter=max_iter)
eval_output['seq_initial'] = gripper_traj

print("single along-with ...")
gripper_traj = diffusion_inference(obj_gt_traj=eval_data['single_along_with']['obj'], gripper_init_T=eval_data['single_along_with']['gripper'], gripper_openness=eval_data["single_along_with"]["openness"], max_iter=max_iter)
eval_output['single_along_with'] = gripper_traj

print("sequence along-with ...")
gripper_traj = diffusion_inference(obj_gt_traj=eval_data['seq_along_with']['obj'], gripper_init_T=eval_data['seq_along_with']['gripper'], gripper_openness=eval_data["seq_along_with"]["openness"], max_iter=max_iter)
eval_output['seq_along_with'] = gripper_traj

# now = datetime.now()
# t = now.strftime("%m.%d.%H.%M")

np.save(f"{log_path}/eval_output-{eval_data_args}.npy", eval_output, allow_pickle=True)

