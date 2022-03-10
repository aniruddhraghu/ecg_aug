### PTB-XL data loading code adapted from https://physionet.org/content/ptb-xl/1.0.0/ ###

PTBXL_PATH = '/path/to/data'

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import random

import pandas as pd
import wfdb
import ast

    
class PTBXL(Dataset):
    def __init__(self, x, y):
        super(PTBXL,self).__init__()
        
        # Downsample to 250 Hz and chop off last 4 samples to get 2496 overall
        if x.shape[1] != 2496 and x.shape[1] == 5000:
            # pad
            x = x[:,::2,:]
            x = x[:,:-4]
        self.x = np.transpose(x, (0,2,1)).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        sample = (x, y)
        return sample


class PTBXLWrapper(object):

    def __init__(self, batch_size, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loaders(self, args):
        
        def load_raw_data(df, sampling_rate, path):
            if sampling_rate == 100:
                data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data

        idxd = {'NORM' : 0, 'MI' : 1, 'STTC' : 2, 'CD' : 3, 'HYP' : 4}
        def aggregate_diagnostic(y_dic):
            tmp = np.zeros(5)
            for key in y_dic.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    tmp[idxd[cls]] = 1
            return tmp

        path = PTBXL_PATH
        sampling_rate=500

        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        y_train = np.stack(y_train, axis=0)
        
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        y_test = np.stack(y_test, axis=0)
        

        # Normalisation: follow PTB-XL demo code. Do zero mean, unit var normalisation across all leads, timesteps, and patients
        meansig = np.mean(X_train.reshape(-1))
        stdsig = np.std(X_train.reshape(-1))
        X_train = (X_train - meansig)/stdsig
        X_test = (X_test - meansig)/stdsig

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        rng = np.random.RandomState(args.seed)
        idxs = np.arange(len(y_train))
        rng.shuffle(idxs)
        
        
        train_samp = int(0.8*args.train_samp)
        val_samp = args.train_samp - train_samp

        train_idxs = idxs[:train_samp]
        val_idxs = idxs[train_samp:train_samp+val_samp]
        
        
        if args.task != 'all':
            task_idx = idxd[args.task]
            prevalence = np.mean(y_train[:,task_idx])
            self.weights = []
            for i in y_train[train_idxs][:,task_idx]:
                if i == 1: 
                    self.weights.append(1-prevalence)
                else:
                    self.weights.append(prevalence)

            ft_train = PTBXL(X_train[train_idxs], y_train[train_idxs][:, task_idx])
            ft_val = PTBXL(X_train[val_idxs], y_train[val_idxs][:, task_idx])
            ft_test = PTBXL(X_test, y_test[:, task_idx])
            
        else:
            ft_train = PTBXL(X_train[train_idxs], y_train[train_idxs])
            ft_val = PTBXL(X_train[val_idxs], y_train[val_idxs])
            ft_test = PTBXL(X_test, y_test)

            

        train_loader = torch.utils.data.DataLoader(dataset=ft_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
        val_loader = torch.utils.data.DataLoader(dataset=ft_val,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=ft_test,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)                                            

        return train_loader, val_loader, test_loader

