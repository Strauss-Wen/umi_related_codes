import numpy as np
import torch
import torch.nn as nn
import sys
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from noise_pred_net import ConditionalUnet1D
import pdb
import os
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


log_path = sys.argv[1]

args = OmegaConf.load(f"{log_path}/config.yaml")
writer = SummaryWriter(log_dir=log_path)


class MoveCupDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        dataset_path = args.data.path
        obs_len = args.lengths.obs_len
        pred_len = args.lengths.pred_len
        act_len = args.lengths.act_len
        train_set_ratio = args.data.train_set_ratio

        self.data_dict = np.load(dataset_path, allow_pickle=True).item()

        if self.data_dict['obs_len'] != obs_len:
            print("obs_len is not equal to the value in the genereted data dict!")
            exit()
        
        if self.data_dict['pred_len'] != pred_len:
            print("pred_len is not equal to the value in the genereted data dict!")
            exit()

        if self.data_dict['act_len'] != act_len:
            print("act_len is not equal to the value in the genereted data dict!")
            exit()


        start_idx = int(len(self.data_dict['episodes'].keys()) * train_set_ratio) + 1
        self.test_idx = list(range(start_idx, len(self.data_dict['episodes'].keys())))


    def __len__(self):
        return len(self.test_idx)

    def __getitem__(self, idx):
        seq = self.data_dict['episodes'][self.test_idx[idx]]
        return seq




obs_len = args.lengths.obs_len
pred_len = args.lengths.pred_len
act_len = args.lengths.act_len

obs_dim = args.dims.obs_dim
action_dim = args.dims.action_dim

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


dataset = MoveCupDatasetTest()


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=args.data.num_workers,
    shuffle=args.data.shuffle,
    # don't kill worker process afte each epoch
    persistent_workers=args.data.persistent_workers
)


# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_len
)


num_diffusion_iters = args.diffusion.num_diffusion_iters

noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule=args.diffusion.beta_schedule,
    # clip output to [-1,1] to improve stability
    clip_sample=args.diffusion.clip_sample,
    # our network predicts noise (instead of denoised action)
    prediction_type=args.diffusion.prediction_type
)

# device transfer
device = torch.device('cuda')

# ============================================ testing ============================================

ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=args.training.ema_power)


state_dict = torch.load(f"{log_path}/final.pth", weights_only=True)
ema.load_state_dict(state_dict)
ema.copy_to(noise_pred_net.parameters())
noise_pred_net.to(device=device)

# -------------------------- debug only --------------------------
# state_dict = torch.load("./ckpt/100_final.pth", weights_only=True)
# # pdb.set_trace()

# ema.load_state_dict(state_dict)
# ema.copy_to(noise_pred_net.parameters())
# noise_pred_net.to(device=device)

# ----------------------------------------------------------------

optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=args.training.lr, weight_decay=args.training.weight_decay)


print("Start testing ...")

test_loss = []
with torch.no_grad():
    with tqdm(dataloader, desc='Episodes', leave=False) as tepisodes:
        episode_idx = 0
        for episode in tepisodes:

            nobs = episode['obs'].to(device).float()     # [B obs_len 12]
            naction = episode['act'].to(device).float()  # [B pred_len 6]

            # pdb.set_trace()
            B = nobs.shape[0]

            obs_cond = nobs[:,:obs_len,:]   # [B, obs_len, 12]
            obs_cond = obs_cond.flatten(start_dim=1)    # [B, obs_len * 12]

            action_pred = torch.randn(naction.shape, device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters, device=device)


            for k in noise_scheduler.timesteps:
                noise_pred = noise_pred_net(
                    sample=action_pred,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                action_pred = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=action_pred
                ).prev_sample


            # pdb.set_trace()
            # L2 loss
            loss = nn.functional.mse_loss(naction, action_pred)
            
            loss_cpu = loss.item()
            tepisodes.set_postfix(loss=loss_cpu)
            test_loss.append(loss_cpu)
            writer.add_scalar('Loss/test', loss_cpu, episode_idx)

            episode_idx += 1

test_loss = np.mean(test_loss)
writer.add_scalar('Loss/test-average', test_loss)
print(test_loss)









