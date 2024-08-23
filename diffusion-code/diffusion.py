import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from noise_pred_net import ConditionalUnet1D
import pdb
import os
from datetime import datetime
from omegaconf import OmegaConf
import shutil
from torch.utils.tensorboard import SummaryWriter



os.makedirs("./ckpt", exist_ok=True)
args = OmegaConf.load("./config.yaml")

now = datetime.now()
t = now.strftime("%m.%d.%H.%M")
writer = SummaryWriter(log_dir=f'./logs/{t}')
shutil.copy("./config.yaml", f"./logs/{t}")


class MoveCupDataset(torch.utils.data.Dataset):
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


        # set the first {train_set_ratio} of the whole dataset as the training set
        self.seq_num = int(len(self.data_dict['episodes'].keys()) * train_set_ratio)


    def __len__(self):
        return self.seq_num

    def __getitem__(self, idx):
        seq = self.data_dict['episodes'][idx]
        return seq




obs_len = args.lengths.obs_len
pred_len = args.lengths.pred_len
act_len = args.lengths.act_len

obs_dim = args.dims.obs_dim
action_dim = args.dims.action_dim

#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16


dataset = MoveCupDataset()


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.data.batch_size,
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



# for this demo, we use DDPMScheduler with 100 diffusion iterations
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
noise_pred_net = noise_pred_net.to(device)

# ============================================ training ============================================
num_epochs = args.training.num_epochs

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=args.training.ema_power)


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

# Cosine LR schedule with linear warmup

lr_scheduler = get_scheduler(
    name=args.training.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.training.num_warmup_steps,
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

                obs_cond = nobs[:,:obs_len,:]   # (B, obs_horizon, 24)
                obs_cond = obs_cond.flatten(start_dim=1)    # (B, obs_horizon * 24)
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()

                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond) # [B pred_len 12]

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                
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
            
        average_epoch_loss = np.mean(epoch_loss)
        tglobal.set_postfix(loss=average_epoch_loss)
        writer.add_scalar('Loss/train', average_epoch_loss, epoch_idx)



ema_noise_pred_net = noise_pred_net

# torch.save(ema.state_dict(), f"./logs/{t}/final-{t}-{obs_len}-{pred_len}-{act_len}.pth")
torch.save(ema.state_dict(), f"./logs/{t}/final.pth")









