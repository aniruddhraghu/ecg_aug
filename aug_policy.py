############ Adapted from  https://github.com/moskomule/dda #############


import random
from copy import deepcopy

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, RelaxedOneHotCategorical

from operations import *


class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations,
                 temperature=0.05,
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self,
                input, y):
        if self.training:
            relaxcat = RelaxedOneHotCategorical(torch.Tensor([0.1]).to(input.device), logits=self._weights)
            wt = relaxcat.rsample()
            op_idx = wt.argmax().detach()
            op_mag = wt[op_idx] / wt[op_idx].detach()
            op_weights = torch.zeros(len(self.operations)).to(input.device)
            op_weights[op_idx] = op_mag
            return torch.stack([op_weights[i]*op(input, y) for i, op in enumerate(self.operations)]).sum(0)
        else:
            return self.operations[Categorical(logits=self._weights).sample()](input, y)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0)



class SubPolicy(nn.Module):
    def __init__(self, sub_policy_stage, operation_count=1):
        super().__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self,input,y):
        for stage in self.stages:
            input = stage(input,y)
        return input


class Policy(nn.Module):
    def __init__(self,operations,num_sub_policies=1,operation_count=2,num_chunks=1,):
        super().__init__()
        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations), operation_count)
                                           for _ in range(num_sub_policies)])
        self.num_sub_policies = num_sub_policies
        self.operation_count = operation_count
        self.num_chunks = num_chunks

    def forward(self,x,y):
        x = self._forward(x,y)
        return x

    def _forward(self,input,y):
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input,y)



def all_ops(learn_mag=True, learn_prob=True):

    return [
        RandTemporalWarp(initial_magnitude=[1.0,1.0], learn_magnitude=learn_mag,learn_probability=learn_prob),
        BaselineWander(learn_magnitude=learn_mag,learn_probability=learn_prob),
        GaussianNoise(learn_magnitude=learn_mag,learn_probability=learn_prob),
        RandCrop(learn_probability=learn_prob),
        RandDisplacement(learn_magnitude=learn_mag,learn_probability=learn_prob),
        MagnitudeScale(learn_magnitude=learn_mag,learn_probability=learn_prob),
        NoOp(),
    ]



def full_policy(num_sub_policies= 1,
                    operation_count= 2,
                    num_chunks= 1,
                    learn_mag=True, learn_prob=True):
    return Policy(nn.ModuleList(all_ops(learn_mag, learn_prob)), num_sub_policies,  operation_count,
                    num_chunks)

