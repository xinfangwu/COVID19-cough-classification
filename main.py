import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm, trange
import importlib

from lib import dataset
from lib.config import config, update_config, infer_exp_id

import wandb 


def evaluate(pred, trg):
    size = len(pred)
    tp = np.count_nonzero(pred * trg)
    tn = np.count_nonzero((1 - pred) * (1 - trg))
    fp = np.count_nonzero((pred) * (1 - trg))
    fn = np.count_nonzero((1 - pred) * (trg))
    acc = (tp + tn) / size
    if (tp+fp)==0:
        precision = tp
    else:
        precision = (tp) / (tp + fp)

    if(tp+fn)==0:
        recall = tp
    else:
        recall = (tp) / (tp + fn)
    
    return acc, precision, recall



def train_loop(model, loader, optimizer):
    model.train()
    compute_loss = nn.CrossEntropyLoss()

    ave_loss, preds, trg = [], [], []

    for i, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        
        # feed forward and compute losses
        outputs = model.forward(batch['img_spec'], batch['img_wave'])
        losses = compute_loss(outputs, batch['target'])

        # backprop
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # evaluation
        _, predicted = outputs.max(1)
        ave_loss.append(losses.item())
        preds.append(predicted.cpu().numpy())
        trg.append(batch['target'].cpu().numpy())
    
    acc, precision, recall = evaluate(np.concatenate(preds),
                                      np.concatenate(trg))

    return {'ave_mean':np.mean(ave_loss),
            'acc': acc,
            'precision': precision,
            'recall': recall}

def valid_loop(model, loader, loss_func):
    model.eval()
    compute_loss = nn.CrossEntropyLoss()

    ave_loss = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 1), position=1, total=len(loader)):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            
            # feed forward and compute losses
            outputs = model.forward(batch['img_spec'], batch['img_wave'])
            losses = compute_loss(outputs, batch['target'])
        

            ave_loss.append(losses.item())
    
    return {'ave_mean':np.mean(ave_loss)}

if __name__ == '__main__':

    wandb.init(project="final-project", entity="cs230-virufy")

    # Parse args & configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="where is the config yaml file for this run")
    args = parser.parse_args()
    update_config(config, args)
    


    wandb.config.update({
            "learning_rate":config.training.lr,
            "batch_size":config.training.batch_size,
            "epoch": config.training.epoch,
            "optim": config.training.optim,
            "model": config.model.modelclass,
            })
    

    # init variable
    exp_dir, exp_id = infer_exp_id(args.cfg, config.ckpt_root)
    os.makedirs(exp_dir, exist_ok=True)
    exp_ckpt_root = os.path.join(exp_dir, exp_id)
    if os.path.isdir(exp_ckpt_root):
        print(f'Warn: the dir {exp_ckpt_root} already existed.')
    else:
        os.makedirs(exp_ckpt_root, exist_ok=True)
    device = 'cuda' if config.cuda else 'cpu'
    if config.cuda and config.cuda_benchmark:
        torch.backends.cudnn.benchmark = True

    # init dataset 
    ## parameters passed from config.py
    DatasetClass = getattr(dataset, config.dataset.name)
    config.dataset.train_kwargs.update(config.dataset.common_kwargs)
    #config.dataset.valid_kwargs.update(config.dataset.common_kwargs)
    config.dataset.test_kwargs.update(config.dataset.common_kwargs)
    ## parameters updated from .yaml  
    train_dataset = DatasetClass(**config.dataset.train_kwargs)
    #valid_dataset = DatasetClass(**config.dataset.valid_kwargs)
    test_dataset = DatasetClass(**config.dataset.test_kwargs)

    # init dataloader
    train_loader = DataLoader(train_dataset, config.training.batch_size,
                                shuffle=True, num_workers=config.num_workers,
                                pin_memory=config.cuda)
    #valid_loader = DataLoader(valid_dataset, 1, num_workers=config.num_workers,
                                #pin_memory=config.cuda)
    test_loader = DataLoader(test_dataset, 1, num_workers=config.num_workers,
                                pin_memory=config.cuda)

    # init network
    model_file = importlib.import_module(config.model.file)
    model_class = getattr(model_file, config.model.modelclass)
    net = model_class(**config.model.kwargs).to(device)
    net.aux_logits = False

    #init optimizer
    if config.training.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                        lr=config.training.lr, 
                        weight_decay=config.training.weight_decay)
    

    for iep in trange(1, config.training.epoch+1, position=0):
        # training 
        ep_loss = train_loop(net, train_loader, optimizer, config.model.loss_func)

        print(f'EP[{iep}/{config.training.epoch}] train:  ' +
              ' \ '.join([f'{k} {v:.3f}' for k, v in ep_loss.items()]))

        # validating
        #ep_loss = valid_loop(net, valid_loader, config.model.loss_func)
        #print(f'EP[{iep}/{config.training.epoch}] valid:  ' +
        #      ' \ '.join([f'{k} {v:.3f}' for k, v in ep_loss.items()]))

        # store the model 
        
        wandb.log({"train_loss": ep_loss['ave_mean'],
            "acc":ep_loss['acc'],
            "precision": ep_loss['precision'],
            "recall":ep_loss['recall']})


        if (iep+1) > 0 and (iep+1)% config.training.save_every == 0:
            pth_name = config.model.modelclass +'/'+ f'ep{iep+1}.pth'
            torch.save(net.state_dict(), os.path.join(config.ckpt_root, pth_name))
            

