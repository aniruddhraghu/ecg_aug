#### Hyperoptimization code adapted from https://github.com/googleinterns/commentaries ####

import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader, Subset
from torch.autograd import grad

# For determinism
from torch.backends import cudnn
cudnn.deterministic = True
cudnn.benchmark = False

from models import *

import aug_policy

from ptbxl_dataset import PTBXLWrapper


import argparse

parser = argparse.ArgumentParser(description='ECG Learn Aug')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--savefol', type=str, default='taskaug')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--hyper_lr', type=float, default=1e-2)
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--aug', default='learnmag')
parser.add_argument('--neumann', type=int, default=1)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--num_base_steps', type=int, default=1)
parser.add_argument('--task',type=str, default='MI')
parser.add_argument('--augckpt',type=str, default=None)
parser.add_argument('--testeval', action='store_true')
parser.add_argument('--only_eval', action='store_true')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.augckpt:
    args.savefol += args.augckpt

SEED=args.seed
torch.manual_seed(SEED)
import random
random.seed(SEED)
np.random.seed(SEED)

dataset_wrapper = PTBXLWrapper(args.batch_size)
train_dataloader, val_dataloader, test_dataloader = dataset_wrapper.get_data_loaders(args)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def model_saver(epoch, student, aug, opt, hyp_opt, path):
    torch.save({
        'epoch' : epoch,
        'aug_sd': aug.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')

    
def get_save_path():
    modfol =  f"""seed{args.seed}-lr{args.lr}-hyperlr{args.hyper_lr}-neumann{args.neumann}-warmup{args.warmup_epochs}-num_base_steps{args.num_base_steps}-trainsamp{args.train_samp}-aug{args.aug}-task{args.task}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth
    

def zero_hypergrad(hyper_params):
    """
    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params


def store_hypergrad(hyper_params, total_d_val_loss_d_lambda):
    """

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for p in hyper_params:
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    i = 0
    while i < num_neumann_terms:  # for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, list(model.parameters()), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term

        preconditioner = preconditioner + counter
        i += 1
    return elementary_lr * preconditioner

def get_hyper_train_flat(hyper_params):
    return torch.cat([p.view(-1) for p in hyper_params])

def gather_flat_grad(loss_grad):
    return torch.cat([p.reshape(-1) for p in loss_grad]) #g_vector

loss_obj = torch.nn.BCEWithLogitsLoss()
def get_loss(enc, x_batch_ecg, y_batch):
    yhat = enc.forward(x_batch_ecg)
    y_batch = y_batch.float()
    loss = loss_obj(yhat.squeeze(), y_batch.squeeze())
    return loss

def hyper_step(model, aug, hyper_params, train_loader, optimizer, val_loader, elementary_lr, neum_steps):
    zero_hypergrad(hyper_params)
    num_weights = sum(p.numel() for p in model.parameters())

    d_train_loss_d_w = torch.zeros(num_weights).to(device)
    model.train(), model.zero_grad()

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        x = do_aug(x, y, aug)
            
        train_loss= get_loss(model, x, y)
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, list(model.parameters()), 
                                                  create_graph=True, allow_unused=True))
        break
    optimizer.zero_grad()

    # Initialize the preconditioner and counter
    # Compute gradients of the validation loss w.r.t. the weights/hypers
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        val_loss = get_loss(model, x, y)
        optimizer.zero_grad()
        d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=False))
        break    
    
    preconditioner = d_val_loss_d_theta

    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,neum_steps, model)

    
    indirect_grad = gather_flat_grad(
        grad(d_train_loss_d_w, hyper_params, grad_outputs=preconditioner.view(-1)))
    hypergrad = indirect_grad # + direct_Grad

    zero_hypergrad(hyper_params)
    store_hypergrad(hyper_params, -hypergrad)
    return hypergrad


# Utility function to update lossdict
def update_lossdict(lossdict, update, action='append'):
    for k in update.keys():
        if action == 'append':
            if k in lossdict:
                lossdict[k].append(update[k])
            else:
                lossdict[k] = [update[k]]
        elif action == 'sum':
            if k in lossdict:
                lossdict[k] += update[k]
            else:
                lossdict[k] = update[k]
        else:
            raise NotImplementedError
    return lossdict


from sklearn.metrics import roc_auc_score, average_precision_score

# Evaluate student on complete train/test set.
def evaluate(dl, student):
    student.eval()
    net_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    ld = {}
    l_obj = nn.BCEWithLogitsLoss(reduction='sum')
    with torch.no_grad():
        for data, target in dl:
            y_true.append(target.detach().cpu().numpy())
            data, target = data.to(device), target.to(device)
            output = student(data)
            net_loss += l_obj(output.squeeze().float(), target.squeeze().float()).item()  # sum up batch loss
            y_pred.append(output.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    net_loss /= len(dl.dataset)
    
    try:
        ld['epoch_loss'] = net_loss
        ld['auc'] = roc_auc_score(y_true, y_pred)
        ld['auprc'] = average_precision_score(y_true, y_pred)
    except ValueError:
        ld['epoch_loss'] = net_loss
        ld['auc'] = 0
        ld['auprc'] = 0

    print(ld)
    return ld


def do_aug(xecg, y, aug):
    xret = aug(xecg,y)
    return xret

def train(train_dl, val_dl, test_dl):
    loss_meter = AverageMeter()
    if args.aug == 'learnmag':
        aug = aug_policy.full_policy(learn_mag=True, learn_prob=False).to(device)
        raise NotImplementedError
    
    hyp_params = list(aug.parameters())
    hyp_optim = torch.optim.RMSprop(hyp_params, lr=args.hyper_lr)
    
    num_outputs = 1
    enc = resnet18(num_outputs=num_outputs).to(device)
    
    optimizer = torch.optim.Adam(enc.parameters(), args.lr)

    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
        load_ep =0
    else:
        ckpt = torch.load(args.checkpoint)
        enc.load_state_dict(ckpt['student_sd'])
        optimizer.load_state_dict(ckpt['optim_sd'])
        hyp_optim.load_state_dict(ckpt['hyp_optim_sd'])
        aug.load_state_dict(ckpt['aug_sd'])
        load_ep = ckpt['epoch'] + 1
        print("Loaded from ckpt")
        
    if args.augckpt:
        ckpt = torch.load(args.augckpt)
        aug.load_state_dict(ckpt['aug_sd'])
        print("Loaded aug")

    if args.only_eval:
        bestmodel = os.path.join(get_save_path(), 'best_model.ckpt')
        enc.load_state_dict(torch.load(bestmodel))
        load_ep = args.epochs

    train_ld = {'loss' : []}
    val_ld = {'loss' : []}
    test_ld = {}
    
    best_val_loss = np.inf
    best_model = copy.deepcopy(enc.state_dict())
    
    num_neumann_steps = args.neumann
    
    steps = 0
    for epoch in range(load_ep,args.epochs):
        for i, (xecg, y) in enumerate(train_dl):
            enc.train()
            zero_hypergrad(hyp_params)
            xecg = xecg.to(device)
            y = y.to(device)
            xecg = do_aug(xecg, y, aug)            
            loss = get_loss(enc, xecg, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch >= args.warmup_epochs and steps % args.num_base_steps == 0 and len(hyp_params) > 0:
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']
                    break

                hypg = hyper_step(enc, aug, hyp_params, train_dl, optimizer, val_dl, cur_lr, num_neumann_steps)
                hypg = hypg.norm().item()
                hyp_optim.step()

            steps += 1
            loss_meter.update(loss.item())
            train_ld['loss'].append(loss.item())


        print("Eval at epoch ", epoch)
        lossdict = evaluate(val_dl, enc)
        val_ld = update_lossdict(val_ld, lossdict)
        cur_val_loss = lossdict['epoch_loss']
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            best_model = copy.deepcopy(enc.state_dict())
        
        tosave = {
            'train_ld' : train_ld,
            'val_ld' : val_ld,
        }
        torch.save(tosave, os.path.join(get_save_path(), 'logs.ckpt'))
        torch.save(best_model, os.path.join(get_save_path(), 'best_model.ckpt'))
        if epoch % 20 == 0 or epoch == args.epochs -1:
            model_saver(epoch, enc, aug, optimizer, hyp_optim, get_save_path())
            print(f"Saved model at epoch {epoch}")
        
    import time
    print(time.time())
    print("Evaluating best model...")
    enc.load_state_dict(best_model)
    lossdict = evaluate(test_dl, enc)
    print(time.time())
    test_ld = update_lossdict(test_ld, lossdict)
    tosave = {
            'train_ld' : train_ld,
            'val_ld' : val_ld,
            'test_ld' : test_ld,
        }
    torch.save(tosave, os.path.join(get_save_path(), 'eval_logs.ckpt'))


print("Checking if run complete")
savepath = os.path.join(get_save_path(), 'eval_logs.ckpt')
if os.path.exists(savepath):
    valaucs = torch.load(savepath)['val_ld']['auc']
    if len(valaucs) == args.epochs:
        print(f"Finished this one {savepath}")
        import sys
        sys.exit(0)

res = train(train_dataloader, val_dataloader, test_dataloader)
