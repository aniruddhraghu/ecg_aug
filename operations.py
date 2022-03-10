############ Adapted from  https://github.com/moskomule/dda #############


""" Operations

"""
from scipy.special import logit

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli

from functional import (rand_temporal_warp, baseline_wander, gaussian_noise, rand_crop, rand_crop_base, spec_aug, rand_displacement, magnitude_scale)

import warp_ops


class _Operation(nn.Module):
    """ Base class of operation

    :param operation:
    :param initial_magnitude:
    :param initial_probability:
    :param learn_magnitude:
    :param learn_probability:
    :param temperature: Temperature for RelaxedBernoulli distribution used during training
    """

    def __init__(self,
                 operation,
                 initial_magnitude,
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__()
        self.operation = operation

        if initial_magnitude is not None and learn_magnitude:
            self.magnitude = nn.Parameter(torch.Tensor(initial_magnitude))
        else:
            self.magnitude = torch.Tensor(initial_magnitude)
        
        if learn_probability:
            self.probability = nn.Parameter(torch.Tensor([float(logit(i)) for i in initial_probability]))
        else:
            self.probability = torch.Tensor([float(logit(i)) for i in initial_probability])

        assert 0 < temperature
        self.temperature = temperature


    def forward(self,input, label):
        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel)
        mask = mask.view(BS, 1, 1)
        retval = (mask * transformed + (1 - mask) * input)
        return retval

    def get_mask(self, label,
                 batch_size=None):

        prob = torch.sigmoid(self.probability).unsqueeze(0)
        prob = prob.repeat(batch_size, 1)
        prob = 0*prob.sum() + prob[torch.arange(batch_size), label.long()]
        if self.training:
            return RelaxedBernoulli(self.temperature, prob).rsample()
        else:
            return Bernoulli(prob).sample()


        
        
class NoOp(_Operation):
    def __init__(self,
                 initial_magnitude=[2., 2.],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature = 0.1,
                ):
        super().__init__(None, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)
        
    def forward(self, input, label):
        return input


class RandTemporalWarp(_Operation):
    def __init__(self,
                 initial_magnitude=[2., 2.],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_temporal_warp, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

        # create the warp obj here.
        self.warp_obj = warp_ops.RandWarpAug([2496])

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        retval = (mask * transformed + (1 - mask) * input)
        return retval



class BaselineWander(_Operation):
    def __init__(self,
                 initial_magnitude=[0.0,0.0],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(baseline_wander, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)


class GaussianNoise(_Operation):
    def __init__(self,
                 initial_magnitude=[0.0,0.0],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(gaussian_noise, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

class RandCrop(_Operation):
    def __init__(self,
                 initial_magnitude=[0.05],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=False,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_crop, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device)
        transformed = self.operation(input, mag)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        retval = (mask * transformed + (1 - mask) * input)
        return retval



class RandDisplacement(_Operation):
    def __init__(self,
                 initial_magnitude=[0.5,0.5],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_displacement, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

    # create the warp obj here.
        self.warp_obj = warp_ops.DispAug([2496])

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS, C, L = input.shape
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B, C, L = transformed.shape
        mask = mask.view(B, 1, 1)
        retval = (mask * transformed + (1 - mask) * input)
        return retval

class MagnitudeScale(_Operation):
    def __init__(self,
                 initial_magnitude=[0.0,0.0],
                 initial_probability=[0.9999999,0.9999999],
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(magnitude_scale, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)


