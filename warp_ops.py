#### Adapted from https://github.com/voxelmorph/voxelmorph ####


import numpy as np 
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli

import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d

KSIZE = 15
PADSIZE = (KSIZE-1) //2
DOWN_FACTOR = 2


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)
#         print("grid shape", grid.shape)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(src.shape) == 3:
            src = src.unsqueeze(-1).repeat(1,1,1,2)
            new_locs = new_locs.unsqueeze(-1).repeat(1,1,1,2)
        
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        
        samp = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return samp.squeeze(2)



class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransformTime(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, sf, ndims):
        super().__init__()
        self.sf = sf
        self.mode = 'linear'

    def forward(self, x):
        factor = self.sf
        if factor < 1:
            x = F.interpolate(x, align_corners=False, scale_factor=factor, mode=self.mode, recompute_scale_factor=False)
            x = factor * x
        elif factor > 1:
            x = factor * x
            x = F.interpolate(x, align_corners=False, scale_factor=factor, mode=self.mode, recompute_scale_factor=False)
        return x


class RandWarpAug(nn.Module):
    def __init__(self, inshape, int_steps = 7, int_downsize = 4, smooth_size = 51):
                 
        super().__init__()
        
        ndims=1
        self.inshape=inshape
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransformTime(1/int_downsize, ndims) if resize else None
        self.fullsize = ResizeTransformTime(int_downsize, ndims) if resize else None

        # configure optional integration layer for diffeomorphic warp
        down_shape = [inshape[0]//int_downsize]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
        
        # set up smoothing filter
        self.smooth_size= smooth_size
        self.smooth_pad = smooth_centre = (smooth_size-1)//2
        smooth_kernel = np.zeros(smooth_size)
        smooth_kernel[smooth_centre] = 1
        filt = gaussian_filter1d(smooth_kernel, smooth_centre).astype(np.float32)
        self.smooth_kernel = torch.from_numpy(filt)

    def forward(self, source, flow_mag):
        x = source

        flow_field = flow_mag*torch.randn(x.shape[0], 1, self.inshape[0]).to(x.device)
        
        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # DO SOME SMOOTHING OF THE FLOW FIELD HERE
        pos_flow = F.conv1d(pos_flow, self.smooth_kernel.view(1,1,self.smooth_size).to(x.device), padding=self.smooth_pad, stride=1)

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        return y_source

class DispAug(nn.Module):
    def __init__(self, inshape, int_steps = 7, int_downsize = 4, flow_mag=1.0, smooth_size = 51, use_label=False):

        super().__init__()
        
        ndims=1
        self.inshape=inshape

        # configure transformer
        self.transformer = SpatialTransformer(inshape)
    

    def forward(self, source, mag):
        BS, C, L = source.shape
        x=source

        disps = (-2*mag)*(torch.rand(BS).to(x.device).view(BS,1,1)) + mag
        # print("dispmag", disps)
        pos_flow = torch.ones(BS, 1, L).to(x.device) * disps.view(BS, 1, 1)
        
        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        return y_source