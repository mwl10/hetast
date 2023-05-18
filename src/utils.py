# pylint: disable=E1101
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from dataset import DataSet, ZtfDataSet
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand
import os
import sys
import model
import torch.optim as optim
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand, gpSimFull
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler
import pickle
import warnings
 

    
## creates an array for kl annealing schedule, credits to: https://github.com/haofuml/cyclical_annealing
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5) -> np.ndarray:
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def save_obj(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp)

def load_obj(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


def save_pickle(data_folder, save_folder):
    lcs = get_data(data_folder)
    save_file = os.path.basename(data_folder) + '.pkl'
    with open(os.path.join(save_folder, save_file), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(lcs, outp, pickle.HIGHEST_PROTOCOL)
    


def get_data(folder, sep=',', start_col=1, batch_size=2, min_length=2, n_union_tp=3500,
             num_resamples=0,shuffle=True,chop=False,test_split=0.1,
             seed=2, keep_missing=True, norm=True):
    
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy') # mean of empty slice
 
    """
    This function provides a way to create & format a dataset for training hetvae. 
    It expects a folder containing folders for each band you would like to add to the dataset.
    The band folders should be labeled as the band name: 'i', 'g', 'u', 'r', etc. The files inside the band should
    be labeled with the object name followed by an underscore. The data files are assumed to be csvs, 
    but if you have tsvs you can set sep='\t'. The data files columns must be in the order of: time, flux/mag, fluxerr/mag/err.
    example directory:
    
    ZTF_DR_data/
    ├── g
         └──  000018.77+191232.9_DR_gband.csv
              000111.81-092508.2_DR_gband.csv
    ├── i
         └──  000018.77+191232.9_DR_iband.csv
              000111.81-092508.2_DR_iband.csv
    
    parameters:
        folder      (str)  --> container folder 
        ----- optional ----- 
        seed        (int)  0    --> random seed for reproducing shuffles & such
        sep         (str)  ","  --> user defined delimiter
        start_col   (int)  0    --> sets the first column in the data files where time points are located, 
                                       assumes column order as time, flux/mag, fluxerr/magerr from there
        batch_size  (int)  8    --> batch size for the training dataloader  
        
    returns:
        the DataSet object we created containing the dictionary of pytorch dataloaders for the network
    """
    np.random.seed(seed=seed) 
    torch.manual_seed(seed=seed)
    if folder.lower().find('ztf') > 0: lcs = ZtfDataSet(folder)
    else: lcs = DataSet(folder)
    lcs.files_to_df()
    lcs.read(sep=sep)
    lcs.prune(min_length=min_length, start_col=start_col, keep_missing=keep_missing)     
    if chop: lcs.chop_lcs() 
    lcs.resample_lcs(num_resamples=num_resamples, seed=seed)
    if norm: lcs.normalize()
    lcs.format_()
    lcs.set_union_tp(uniform=True,n=n_union_tp)
    print(f'dataset created, {lcs.dataset.shape=}')
    lcs.set_data_obj(batch_size=batch_size, shuffle=shuffle,test_split=test_split, seed=seed)
    return lcs

            

    
# old...    
# def load_checkpoint(filename, data_obj, device='mps'):
#     """
#     loads a model checkpoint 
#     """
#     if os.path.isfile(filename):
#         print("=> loading checkpoint '{}'".format(filename))
#         cp = torch.load(filename, map_location=torch.device(device))
#         cp['args'].device = device
#         cp['args'].net = 'HeTVAE'
#         cp['args'].num_heads = cp['args'].enc_num_heads
#         print(cp['args'])
#         net = model.load_network(cp['args'], data_obj['input_dim'], data_obj['union_tp'])
#         net.load_state_dict(cp['state_dict'])
#         params = list(net.parameters())
#         optimizer = optim.Adam(params)
#         optimizer.load_state_dict(cp['optimizer_state_dict'])
#         return net,optimizer, cp['args'], cp['epoch'], cp['loss'], cp['train_losses'], cp['test_losses']
    
def load_checkpoint(filename, data_obj, device='mps'):
    """
    loads a model checkpoint 
    """
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        cp = torch.load(filename, map_location=torch.device(device))
        cp['args'].device = device
        print(cp['args'])
        net = model.load_network(cp['args'], data_obj['input_dim'], data_obj['union_tp'])
        net.load_state_dict(cp['state_dict'])
        params = list(net.parameters())
        optimizer = optim.Adam(params)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        return net,optimizer,scheduler,cp['lrs'], cp['args'], cp['epoch'], cp['losses']
    
    

def make_masks(batch, frac=0.5, forecast=False) -> torch.Tensor:
    """
    method to subsample a fraction of observed points in a light curve for
    training
    
    parameters:
        batch (torch.Tensor)
        ----- optional -----
        frac  (float)  0.5  --> 0 to 1, dictates the fraction of points we mask off for training
    """
    subsampled_mask = torch.zeros_like(batch[:,:, :, 1])
    for i, object_lcs in enumerate(batch):
        for j, lc in enumerate(object_lcs):
            indexes = lc[:,1].nonzero()[:,0] # where the observations are
            length = int(len(indexes) * frac)
            if forecast:
                # take first section of the points explicitly
                subsampled_points = indexes[:length]
            else:
                subsampled_points = np.sort(np.random.choice(indexes, \
                                                  size=length, \
                                                  replace=False))
            # set the mask at the subsampled points
            subsampled_mask[i,j,subsampled_points] = 1
   
    return subsampled_mask


def update_lr(model_size, itr, warmup):
    """
    update the learning rate according to the scheduler from astromer (Donoso-Oliva et al. 2022)
    """
    lr = (model_size ** -0.5) * min(itr**-0.5, itr * warmup**-0.5)
    return lr


def evaluate_hetvae(
    net,
    dim,
    dataloader,
    frac=0.5,
    k_iwae=1,
    device='mps',
    forecast=False,
    qz_mean=False
):
    train_n = 0
    train_loss,avg_loglik, mse, mae = 0, 0, 0,0
    mean_mae, mean_mse = 0, 0
    individual_nlls = []
    indy_nlls = []
    mses= []
    with torch.no_grad():
        for batch in dataloader:
            batch_len = batch.shape[0]
            # forecasting if this mask is set to first section of points only, not random sub-selection
            subsampled_mask = make_masks(batch, frac=frac, forecast=forecast)
            ######################
            errorbars = torch.swapaxes(batch[:,:,:,2], 2,1)
            weights = errorbars.clone()
            weights[weights!=0] = 1 / weights[weights!=0]
            errorbars[errorbars!=0] = torch.log(errorbars[errorbars!=0])
            logerr = errorbars.to(device)
            weights = weights.to(device)
            ######################
            batch = batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(subsampled_mask, batch[:,:,:,1])

            context_y = torch.cat((
              batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1)
            recon_context_y = torch.cat((
              batch[:,:,:,1] * recon_mask, recon_mask
            ), 1).transpose(2,1)
            
            loss_info = net.compute_unsupervised_loss(
              batch[:, 0, :,0],
              context_y,
              batch[:, 0, :,0],
              recon_context_y,
              logerr,
              weights,
              num_samples=k_iwae,
              qz_mean=qz_mean
            )
            
            individual_nlls.append(loss_info.loglik_per_ex.cpu().numpy())
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            mse += loss_info.mse * batch_len
            train_n += batch_len
            

    avg_nll =  -avg_loglik / train_n
    avg_mse = mse / train_n
    
#     print(
#         'nll: {:.4f}, mse: {:.4f}'.format(
#             avg_nll,
#             avg_mse)
#     )

    return avg_nll, avg_mse, -1 * np.concatenate(individual_nlls, axis=1)[0]




def encode(dataloader, net, device='mps',subsample=False):
    qz_mean, qz_std = [], []
    disc_path = [] ## deterministic/discrete pathway
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if subsample == True:
                subsampled_mask = make_masks(batch, frac=0.5)
            else:
                subsampled_mask = make_masks(batch, frac=1.0)
            batch = batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            context_y = torch.cat((batch[:,:,:,1] * subsampled_mask, subsampled_mask), \
                                  1).transpose(2,1)
            qz, hidden = net.encode(batch[:, 0, :,0], context_y)
            hidden = torch.cat((hidden[:, :, :, 0], hidden[:, :, :, 1]), -1)
            hidden = net.proj(hidden)
            qz_mean.append(qz.mean.cpu().numpy())
            qz_std.append(torch.exp(0.5 * qz.logvar).cpu().numpy())
            disc_path.append(hidden.cpu().numpy())
            
    qz_mean = np.concatenate(qz_mean, axis=0)
    qz_std = np.concatenate(qz_std, axis=0)
    disc_path = np.concatenate(disc_path, axis=0) 
    qzs = np.concatenate((qz_mean[:,np.newaxis], qz_std[:,np.newaxis]), axis=1)
    return qzs,disc_path



def decode(net,zs,disc_path,target_x,device='mps',batch_size=2):
    z = np.concatenate((zs,disc_path),axis=-1).astype(np.float32) # 234,16,128, 
    dl = torch.utils.data.DataLoader(z, batch_size=batch_size, shuffle=False)
    pred_mean, pred_std = [], []
    target_tp = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            batch = batch.to(device)
            tx = torch.tensor(target_x[i*batch_size:(i*batch_size + batch_size)])[:,0]
            tx = tx.to(device)
            # zbatch shape should be (1, batch_size,n ref pts, latent_dim * 2)
            px = net.decode(batch.unsqueeze(0), tx) 
            pred_mean.append(px.mean.cpu().numpy())
            pred_std.append(torch.exp(0.5 * px.logvar).cpu().numpy())
            target_tp.append(tx.cpu().numpy())
    pred_mean = np.concatenate(pred_mean, axis=1)
    pred_std = np.concatenate(pred_std, axis=1)
    target_tp = np.concatenate(target_tp, axis=0)
    interps = np.zeros((target_x.shape[0],target_x.shape[1], target_x.shape[2],3))
    interps[:,:,:,0] = target_tp[:,np.newaxis].repeat(target_x.shape[1],axis=1)
    interps[:,:,:,1] = pred_mean.squeeze(0).swapaxes(1,2)
    interps[:,:,:,2] = pred_std.squeeze(0).swapaxes(1,2)
    return interps
                  
    
def predict(dataloader, net,target_x=None,device='mps',forecast=False,k_iwae=1,frac=0.5,qz_mean=False):
    pred_mean, pred_std = [], []
    qz_mu, qz_std = [], []
    mask = []
    target = []
    tp =[]
    target_tp = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            batch_len = batch.shape[0] 
            subsampled_mask = make_masks(batch, frac=frac, forecast=forecast)
            batch = batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(batch[:,:,:,1], subsampled_mask)
            context_y = torch.cat((batch[:,:,:,1] * subsampled_mask, subsampled_mask), 1).transpose(2,1)
            recon_context_y=torch.cat((batch[:,:,:,1] * recon_mask, recon_mask), 1).transpose(2,1)
            
            if target_x is not None:
                tx = torch.tensor(target_x[i*batch_len:(i*batch_len + batch_len)])[:,0]
            else:
                tx = batch[:, 0, :,0]
                
            px, qz = net.get_reconstruction(batch[:, 0, :,0], context_y, tx, num_samples=k_iwae,qz_mean=qz_mean)
            
            pred_mean.append(px.mean.cpu().numpy())
            pred_std.append(torch.exp(0.5 * px.logvar).cpu().numpy())
            qz_mu.append(qz.mean.cpu().numpy())
            qz_std.append(torch.exp(0.5 * qz.logvar).cpu().numpy())
            target.append((batch[:, :, :,1]).cpu().numpy()) # only targets when we don't have a specific target_x
            mask.append(subsampled_mask.cpu().numpy()) 
            tp.append(batch[:, 0, :,0].cpu().numpy())
            target_tp.append(tx.cpu().numpy())       
    # reconstructions
    pred_mean = np.concatenate(pred_mean, axis=1)
    pred_std = np.concatenate(pred_std, axis=1)
    # latent space 
    qz_mu = np.concatenate(qz_mu, axis=0)
    qz_std = np.concatenate(qz_std, axis=0)
    # targets are all ys, masks are masks, inputs are masked ys
    target = np.concatenate(target, axis=0)
    mask = np.concatenate(mask, axis=0)
    inputs = np.ma.masked_where(mask < 1., target)
    inputs = np.transpose(inputs, [0,2,1])
    target = np.transpose(target, [0,2,1])
    mask = np.transpose(mask, [0,2,1])
    # target tp are points we'd like to project to
    target_tp = np.concatenate(target_tp,axis=0)
    tp = np.concatenate(tp, axis=0)
    pred_mean = pred_mean.mean(0)
    pred_std = pred_std.mean(0)
    
    recons = {'target_tp':target_tp, 'pred_mean':pred_mean,'pred_std':pred_std}
    zs = {'qz_mu':qz_mu, 'qz_std':qz_std}
    examples = {'tp':tp,'target':target,'mask':mask,'inputs':inputs}
    recon_info = {'examples':examples,'recons':recons, 'zs':zs}
    return recon_info


def plot_recons(examples, recons, N=7, figsize=(35,5)):
    target_tp = recons['target_tp']
    pred_mean = recons['pred_mean']
    pred_std = recons['pred_std']
    
    target = examples['target']
    inputs = examples['inputs']
    tp = examples['tp']
    print(target_tp.shape, pred_mean)
    dims = pred_mean.shape[2]
    fig,ax = plt.subplots(N,dims,figsize=figsize, squeeze=False)
    for ex in range(N):
        pred_t = target_tp[ex].nonzero()[0]
        input_t = tp[ex].nonzero()[0]
        for band in range(dims):
            std = pred_std[ex,pred_t,band]
            # preds plotting
            ax[ex,band].plot(target_tp[ex, pred_t],pred_mean[ex,pred_t,band])
            ax[ex,band].fill_between(target_tp[ex,pred_t],pred_mean[ex,pred_t,band]-std \
                                     ,pred_mean[ex,pred_t,band]+std, label='error envelope' \
                                     ,color='lightcoral')
            ax[ex,band].errorbar(target_tp[ex,pred_t], pred_mean[ex,pred_t,band], \
                                 yerr=std, c='blue', ecolor='#65c9f7', label='prediction')
            # inputs the preds are conditioned on
            ax[ex,band].scatter(tp[ex,input_t], inputs[ex,input_t,band], \
                                c='black', marker='x', zorder=30, label='conditioned on', s=25)
            #ax[ex,band].set_ylim([-2,2])
            
            

# def preview_lcs(lcs, n=1, figsize=(15,15), fs=15):
#     dims = lcs.dataset.shape[1]
#     fig, ax = plt.subplots(n, dims, figsize=figsize, squeeze=False)
#     fig.tight_layout(pad=5.0)
#     for i in range(n):
#         obj_name = lcs.valid_files_df.index.values[i]
#         ax[i][0].set_title(obj_name,fontsize=fs)
#         for band in range(dims):
#             t = lcs.dataset[i,band,:,0]
#             y = lcs.dataset[i,band,:,1]
#             yerr = lcs.dataset[i,band,:,2]
#             pts = y.nonzero()[0]
#             ax[i][band].errorbar(t[pts],y[pts], yerr=yerr[pts], c='blue', fmt='.', \
#                                  markersize=4, ecolor='red', elinewidth=1, capsize=2)
#     lines_labels = ax[0][0].get_legend_handles_labels()
#     lines,labels = lines_labels[0], lines_labels[1]
#     fig.legend(lines, labels, loc='upper left')
#     [ax[i][index].set_xlabel(lcs.bands[index],fontsize=fs+10) for index in range(len(lcs.bands))]
    

def preview_lcs(lcs, indexes=[0,1,34,100], figsize=(15,15), fs=30):
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    dims = lcs.dataset.shape[1]
    fig, ax = plt.subplots(len(indexes), dims, figsize=figsize, squeeze=False)
    fig.tight_layout(pad=5.0)
    c = 0
    for i in indexes:
        obj_name = lcs.valid_files_df.index.values[i]
        ax[c][0].set_title(obj_name,fontsize=fs)
        for band in range(dims):
            t = lcs.dataset[i,band,:,0]
            y = lcs.dataset[i,band,:,1]
            yerr = lcs.dataset[i,band,:,2]
            pts = y.nonzero()[0]
            ax[c][band].errorbar(t[pts],y[pts], yerr=yerr[pts], c='blue', fmt='.', markersize=4, 
                                 ecolor='red', elinewidth=1, capsize=2)
        c+= 1
    lines_labels = ax[0][0].get_legend_handles_labels()
    lines,labels = lines_labels[0], lines_labels[1]
    fig.legend(lines, labels, loc='upper left')
    [ax[c-1][index].set_xlabel(lcs.bands[index],fontsize=fs+10) for index in range(len(lcs.bands))]
    
    
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_normal_pdf(x, mean, logvar, mask, logerr=0.):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    logerr = logerr * mask
    logvar = logvar + logerr
    return -0.5 * (const + torch.log(torch.exp(logerr) + torch.exp(logvar)) \
            +  (x - mean) ** 2.0 / (torch.exp(logvar)+torch.exp(logerr))) * mask
    
    #return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask


def mog_log_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    const2 = torch.from_numpy(np.array([mean.size(0)])).float().to(x.device)
    loglik = -0.5 * (const + logvar + (x - mean) **
                     2.0 / torch.exp(logvar)) * mask

    return torch.logsumexp(loglik - torch.log(const2), 0)

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl

def mean_squared_error(orig, pred, mask, weights=1.):
    weights = weights * mask
    error = ((orig - pred) ** 2) * weights
    error = error * mask
    return error.sum() / mask.sum()

def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()

import omegaconf
def get_leaf_nodes(cfg: omegaconf.DictConfig):
    leaf_nodes = {}
    for key, value in cfg.items():
        if isinstance(value, omegaconf.DictConfig):
            leaf_nodes.update(get_leaf_nodes(value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, omegaconf.DictConfig):
                    leaf_nodes.update(get_leaf_nodes(item))
                else:
                    leaf_nodes[f"{key}[{i}]"] = item
        else:
            leaf_nodes[key] = value
    return leaf_nodes
