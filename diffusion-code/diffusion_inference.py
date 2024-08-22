import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from noise_pred_net import ConditionalUnet1D
import pdb
import sys
from diffusers.training_utils import EMAModel
from tqdm import tqdm
from omegaconf import OmegaConf

conf = OmegaConf.load('./config.yaml')

obs_len = conf.lengths.obs_len # 2
pred_len = conf.lengths.pred_len # 16
act_len = conf.lengths.act_len # 8
obs_dim = conf.dims.obs_dim # 12
action_dim = conf.dims.action_dim # 6
num_diffusion_iters = conf.diffusion.num_diffusion_iters # 100

device = torch.device('cuda')

# TODO: fix dimensions for observations and actions here (12, 6 respectively)

ema_noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_len
    )
ema = EMAModel(
    parameters=ema_noise_pred_net.parameters(),
    power=0.75)

state_dict = torch.load("./ckpt/100_final.pth", weights_only=True)
# pdb.set_trace()

ema.load_state_dict(state_dict)
ema.copy_to(ema_noise_pred_net.parameters())
ema_noise_pred_net.to(device=device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)



def diffusion_eval(cup_init_T: np.ndarray, gripper_init_T: np.ndarray, max_iter: int):
    # cup_init_T: [4 4] only use the first frame pose
    # gripper_init_T: [4 4] same as above

    B = 1

    # normalize observation
    if (len(cup_init_T.shape)) == 2:
        # [4 4]
        obs_cup = np.zeros(shape=[obs_len, obs_dim // 2])
        obs_gripper = gripper_init_T - cup_init_T
        # obs_gripper = obs_gripper[:3]   # [3 4]
        obs_gripper = np.repeat(obs_gripper[np.newaxis], repeats=obs_len, axis=0).reshape([obs_len, 12])   # [obs_len 12]
    
    else:
        # [N 4 4]
        obs_cup = cup_init_T[:obs_len, :3]  # [obs_len 3 4]
        obs_cup = obs_cup - obs_cup[0]
        obs_cup = obs_cup.reshape([obs_len, -1])

        obs_gripper = gripper_init_T - cup_init_T[0]
        obs_gripper = obs_gripper[:obs_len, :3]    # [obs_len 3 4]
        obs_gripper = obs_gripper.reshape([obs_len, -1])    # [obs_len 12]


    gripper_traj = [obs_gripper[i] for i in range(obs_len)] # [N [12]]

    # pdb.set_trace()

    with torch.no_grad():
        for iter_num in tqdm(range(max_iter), desc="iter", miniters=1):
            
            # assume the cup can't move, because there is no simulator for acquiring the cup's trajectory
            obs_i_cup = obs_cup
            obs_i_gripper = np.array(gripper_traj[-obs_len:])   # [obs_len 12]
            obs_i = np.concatenate([obs_i_cup, obs_i_gripper], axis=1)  # [obs_len 24]
            obs_i = torch.from_numpy(obs_i).to(device=device, dtype=torch.float32)

            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_i.unsqueeze(0) # [1 obs_len 24]
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


            naction = naction[0].detach().to('cpu').numpy()    # [pred_len 12]
            action = naction[:act_len]  # [act_len 12]

            # execute action_horizon number of steps without replanning
            for i in range(len(action)):
                # pdb.set_trace()
                gripper_pose_i = gripper_traj[-1] + action[i]
                gripper_traj.append(gripper_pose_i)


    # pdb.set_trace()

    # pad/reshape from [N 12] to [N 4 4]
    frame_num = len(gripper_traj)
    gripper_traj = np.array(gripper_traj)   # [N 12]
    gripper_traj = gripper_traj.reshape([frame_num, 3, 4])
    pad = np.array([0, 0, 0, 1])    # [4]
    pad = np.repeat(pad[np.newaxis], repeats=frame_num, axis=0) # [N 4]
    pad = pad[:, np.newaxis, :] # [N 1 4]
    gripper_traj = np.concatenate([gripper_traj, pad], axis=1)

    if len(cup_init_T.shape) == 2:
        gripper_traj += cup_init_T
    else:
        gripper_traj += cup_init_T[0]


    return gripper_traj

if __name__ == "__main__":
    eval_path = sys.argv[1]
    scene_name = eval_path.split("/")[-1].split(".")[0]
    print(scene_name)
    eval_data = np.load(eval_path, allow_pickle=True).item()

    eval_output = {}
    max_iter = 10
    

    print("single initial ...")
    gripper_traj = diffusion_eval(cup_init_T=eval_data['single_initial']['cup'], gripper_init_T=eval_data['single_initial']['gripper'], max_iter=max_iter)
    eval_output['single_initial'] = gripper_traj

    print("sequence initial ...")
    gripper_traj = diffusion_eval(cup_init_T=eval_data['seq_initial']['cup'], gripper_init_T=eval_data['seq_initial']['gripper'], max_iter=max_iter)
    eval_output['seq_initial'] = gripper_traj

    print("single along-with ...")
    gripper_traj = diffusion_eval(cup_init_T=eval_data['single_along_with']['cup'], gripper_init_T=eval_data['single_along_with']['gripper'], max_iter=max_iter)
    eval_output['single_along_with'] = gripper_traj

    print("sequence along-with ...")
    gripper_traj = diffusion_eval(cup_init_T=eval_data['seq_along_with']['cup'], gripper_init_T=eval_data['seq_along_with']['gripper'], max_iter=max_iter)
    eval_output['seq_along_with'] = gripper_traj

    
    np.save(f"./eval_output/eval_{scene_name}.npy", eval_output, allow_pickle=True)
