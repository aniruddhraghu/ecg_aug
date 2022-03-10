import numpy as np
from torch.nn import functional as F

import torch
from torch.autograd import Function



def rand_temporal_warp(x, mag, warp_obj):
    mag = 100*(mag**2)
    return warp_obj(x, mag)

def baseline_wander(x, mag):
    BS, C, L = x.shape

    # form baseline drift
    strength = 0.25*torch.sigmoid(mag) * (torch.rand(BS).to(x.device).view(BS,1,1))
    strength = strength.view(BS, 1, 1)

    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)  # typical breaths per second for an adult
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    drift = strength*torch.sin(torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()).to(x.device)
    return x + drift

def gaussian_noise(x, mag):
    BS, C, L = x.shape
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.25*stdval*torch.sigmoid(mag)*torch.randn(BS, C, L).to(x.device)
    return x + noise

def rand_crop(x, mag):
    x_aug = x.clone()
    # get shapes
    BS, C, L = x.shape
    mag = mag.item()

    nmf = int(mag*L)
    start = torch.randint(0, L-nmf,[1]).long()
    end = (start + nmf).long()
    x_aug[:, :, start:end] = 0.
    return x_aug


def spec_aug(x, mag):
    num_ch = 12
    x_aug = x.clone()
    BS, C, L = x.shape
    mag = mag.item()
    
    # get shapes
    BS, NF, NT, _ = torch.stft(x[:,0,], n_fft=512, hop_length=4).shape
    nmf = int(mag*NF)
    start = torch.randint(0, NF-nmf,[1]).long()
    end = (start + nmf).long()
    for i in range(12):
        stft_inp = torch.stft(x[:,i,], n_fft=512, hop_length=4)
        idxs = torch.zeros(*stft_inp.shape).bool()
        stft_inp[torch.arange(BS).long(), start:end,:] = 0
        x_aug[:, i] = torch.istft(stft_inp, n_fft=512, hop_length=4)
    
   

    nmf = int(mag*L)
    start = torch.randint(0, L-nmf,[1]).long()
    end = (start + nmf).long()
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.
    x_aug[:, :, start:end] = noise 
    return x_aug



def rand_displacement(x, mag, warp_obj):
    disp_mag = 100*(mag**2)
    return warp_obj(x, disp_mag)

def magnitude_scale(x, mag):
    BS, C, L = x.shape
    strength = torch.sigmoid(mag)*(-0.5 * (torch.rand(BS).to(x.device)).view(BS,1,1) + 1.25)
    strength = strength.view(BS, 1, 1)
    return x*strength



