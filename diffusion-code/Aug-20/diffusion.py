import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
# from get_data_dict import get_data_dict
from noise_pred_net import ConditionalUnet1D
import pdb
import os

os.makedirs("./ckpt", exist_ok=True)


# dataset
class MoveCupDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, obs_len, pred_len, act_len):
        # self.data_dict = get_data_dict(demos_root=dataset_path, obs_len=obs_len, pred_len=pred_len, act_len=act_len)
        self.data_dict = np.load(dataset_path, allow_pickle=True).item()

        self.seq_num = len(self.data_dict.keys()) - 2   # except 'obs_range' and 'act_range'

    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):
        seq = self.data_dict[idx]
        return seq



obs_len = 4
pred_len = 8
act_len = 8

obs_dim = 12
action_dim = 6

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


dataset = MoveCupDataset(
    # dataset_path='/mnt/data/collected_demos_xarm',
    # dataset_path='/haozhe-pv/mlp/data_xarm.npy',
    dataset_path='/haozhe-pv/diffusion/data_xarm-normalized.npy',
    obs_len=obs_len,
    pred_len=pred_len,
    act_len=act_len
)


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=1,
    shuffle=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)


# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_len
)



# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
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

# device transfer
device = torch.device('cuda')
noise_pred_net = noise_pred_net.to(device)

# ============================================ training ============================================
num_epochs = 100

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)


# -------------------------- debug only --------------------------
# state_dict = torch.load("./ckpt/100_final.pth", weights_only=True)
# # pdb.set_trace()

# ema.load_state_dict(state_dict)
# ema.copy_to(noise_pred_net.parameters())
# noise_pred_net.to(device=device)

# ----------------------------------------------------------------
# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

print("Start training ...")


with tqdm(range(num_epochs), desc='Epoch') as tglobal:

    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # pdb.set_trace()

                nobs = nbatch['obs'].to(device).float()     # [B obs_len 24]
                naction = nbatch['act'].to(device).float()  # [B pred_len 12]

                # pdb.set_trace()
                B = nobs.shape[0]

                # observation as FiLM conditioning
                obs_cond = nobs[:,:obs_len,:]   # (B, obs_horizon, 24)
                obs_cond = obs_cond.flatten(start_dim=1)    # (B, obs_horizon * 24)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()


                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                # predict the noise residual
                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond) # [B pred_len 12]

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                
                # pdb.set_trace()

                # tensor([-0.0733, -0.9441,  0.4090,  1.8552,  0.0480, -0.4946, -0.1798, -0.5963],
                # tensor([ 0.8655, -0.8500,  0.3172,  0.6180,  1.0778, -0.4532, -0.1724, -0.5449],

                # tensor([-0.8009,  0.7625,  0.5384, -0.0917,  0.1299,  0.7196,  0.4508, -2.0742],
                # tensor([-0.7789,  0.8826,  0.4912, -0.1030,  0.1195,  0.6787,  0.4831, -2.1221],

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()

                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
            

        tglobal.set_postfix(loss=np.mean(epoch_loss))



# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
# ema.copy_to(ema_noise_pred_net.parameters())

torch.save(ema.state_dict(), f"./ckpt/final.pth")










