import copy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader, Subset

from torch.backends import cudnn
cudnn.deterministic = True
cudnn.benchmark = False

from models import *

from ptbxl_dataset import PTBXLWrapper

import argparse

parser = argparse.ArgumentParser(description='ECG Aug Baseline')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--savefol', type=str, default='baseline')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--train_samp', type=int, default=1000)
parser.add_argument('--task',type=str, default='MI')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def model_saver(epoch, student, opt, path):
    torch.save({
        'epoch' : epoch,
        'student_sd': student.state_dict(),
        'optim_sd': opt.state_dict(),
    }, path + f'/checkpoint_epoch{epoch}.pt')

def get_save_path():
    modfol =  f"""seed{args.seed}-lr{args.lr}-trainsamp{args.train_samp}-task{args.task}"""
    pth = os.path.join(args.savefol, modfol)
    os.makedirs(pth, exist_ok=True)
    return pth

loss_obj = torch.nn.BCEWithLogitsLoss()
def get_loss(enc, x_batch_ecg, y_batch):
    yhat = enc.forward(x_batch_ecg)
    y_batch = y_batch.float()
    loss = loss_obj(yhat.squeeze(), y_batch.squeeze())
    return loss

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

def get_preds(dl, enc):
    y_preds = []
    y_trues = []
    enc.eval()
    for i, (xecg, y) in enumerate(dl):
        y_trues.append(y.detach().numpy())
        xecg = xecg.to(device)
        y_pred = enc.forward(xecg)
        y_preds.append(y_pred.cpu().detach().numpy())
    
    return (np.concatenate(y_preds,axis=0), np.concatenate(y_trues,axis=0))

def evaluate(dl, enc):
    enc.eval()
    ld = {}
    loss = 0
    loss_obj = torch.nn.BCEWithLogitsLoss()
    y_preds = []
    y_trues = []
    pbar = dl
    with torch.no_grad():
        for i, (xecg, y) in enumerate(pbar):
            y_trues.append(y.detach().numpy())

            xecg = xecg.to(device)
            y = y.to(device)

            y_pred = enc.forward(xecg)
            y_preds.append(y_pred.cpu().detach().numpy())

            l = loss_obj(y_pred.squeeze(), y.squeeze().float())
            loss += l.item()
    loss /= len(dl)
    (y_preds, y_trues) = (np.concatenate(y_preds,axis=0), np.concatenate(y_trues,axis=0))
    y_preds = np.squeeze(y_preds)
    y_trues = np.squeeze(y_trues)

    try:
        ld['epoch_loss'] = loss
        ld['auc'] = roc_auc_score(y_trues, y_preds, average=None)
        ld['auprc'] = average_precision_score(y_trues, y_preds, average=None)
    except ValueError:
        ld['epoch_loss'] = loss
        ld['auc'] = 0
        ld['auprc'] = 0
    print(ld)
    return ld


def train(train_dl, val_dl, test_dl, warp_aug=None):
    loss_meter = AverageMeter()
    num_outputs = 1
    enc = resnet18(num_outputs=num_outputs).to(device)
    
    optimizer = torch.optim.Adam(enc.parameters(), args.lr)

    if args.checkpoint is None:
        print("No checkpoint! Training from scratch")
        load_ep =0
    else:
        ckpt = torch.load(args.checkpoint)
        student.load_state_dict(ckpt['student_sd'])
        optimizer.load_state_dict(ckpt['optim_sd'])
        load_ep = ckpt['epoch'] + 1
        print("Loaded from ckpt")

    train_ld = {'loss' : []}
    val_ld = {}
    test_ld = {}
    
    print("Checking if run complete")
    savepath = os.path.join(get_save_path(), 'eval_logs.ckpt')
    if os.path.exists(savepath):
        valaucs = torch.load(savepath)['val_ld']['auc']
        if len(valaucs) == args.epochs:
            print(f"Finished this one {savepath}")
            return
    
    best_val_loss = np.inf
    best_model = copy.deepcopy(enc.state_dict())
    
    for epoch in range(load_ep, args.epochs):
        for i, (xecg, y) in enumerate(train_dl):
            enc.train()
            xecg = xecg.to(device)
            y = y.to(device)

            loss = get_loss(enc, xecg, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        torch.save(tosave, os.path.join(get_save_path(), 'eval_logs.ckpt'))
        torch.save(best_model, os.path.join(get_save_path(), 'best_model.ckpt'))
        loss_meter.reset()
    
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
