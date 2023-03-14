# pylint: disable=E1101
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from dataset import DataSet
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand
import os
import sys
import model
import torch.optim as optim
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand, gpSimFull


## creates an array for kl annealing schedule, credits to: https://github.com/haofuml/cyclical_annealing
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
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


def get_data(folder, sep=',', start_col=1, batch_size=8, min_length=1, n_union_tp=3500, num_resamples=0,shuffle=True, extend=0, chop=False, norm_t=False, correct_z=False):
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

    if not os.path.isdir(folder):
        raise Exception(f"{folder} is not a directory")
    ##################################
    # initializing the DataSet objects 
    ##################################
    lcs = DataSet(folder, sep=sep, start_col=start_col)
    band_folders = os.listdir(folder)
    for band_folder in band_folders:
        lcs.add_band(band_folder, os.path.join(folder, band_folder))
    ### preprocessing functions ####################################################################
    lcs.filter(min_length=min_length)            # filter short light curves; points w/ bad catflags 
    lcs.prune_outliers()    # filter points outside 10 std
    if chop: lcs.chop_lcs() # points past 1 std beyon mean of lengths
    lcs.resample_lcs(num_resamples=num_resamples)
    if correct_z: lcs.correct_z()
    lcs.normalize(norm_t=norm_t)
    lcs.format(extend=extend)
    lcs.set_union_tp(uniform=True,n=n_union_tp)
    print(f'dataset created w/ shape {lcs.dataset.shape}')
    ######## done preprocessing ######################################################################
    lcs.set_data_obj(batch_size=batch_size, shuffle=shuffle)
    return lcs

        
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
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        return net,optimizer, cp['args'], cp['epoch'], cp['loss'], cp['train_losses'], cp['test_losses']
    

def make_masks(batch, frac=0.5, forecast=False):
    """
    method to subsample a fraction of observed points in a light curve for
    hetvae's unsupervised training
    
    parameters:
        batch (torch.Tensor)
        ----- optional -----
        frac  (float)  0.5  --> 0 to 1, dictates the fraction of points we mask off for training
    """
    subsampled_mask = torch.zeros_like(batch[:,:, :, 1])
    recon_mask = torch.zeros_like(batch[:, :,:, 1])
    for i, object_lcs in enumerate(batch):
        for j, lc in enumerate(object_lcs):
            ############################
            # where the observations are
            ############################
            indexes = lc[:,1].nonzero()[:,0] 
            ############################
            # take a fraction of those
            ############################
            length = int(np.round(len(indexes) * frac)) 
            subsampled_points = np.sort(np.random.choice(indexes, \
                                                  size=length, \
                                                  replace=False))
#             if forecast:
#                 # want ones at all points less than 1550.2127838134766 and nonzero
#                 subsampled_points = np.intersect1d(indexes, np.where(lc[:,0] < 1550.2127838134766)[0])
            ############################
            # set the mask at the subsampled points
            ############################
            subsampled_mask[i,j,subsampled_points] = 1
   
    return subsampled_mask

    
def update_lr(model_size, itr, warmup):
    """
    helper function to update the learning rate according to the scheduler
    """
    lr = (model_size ** -0.5) * min(itr**-0.5, itr * warmup**-0.5)
    return lr


def evaluate_hetvae(
    net,
    dim,
    train_loader,
    frac=0.5,
    k_iwae=1,
    device='cuda',
    forecast=False
):
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)
    train_n = 0
    train_loss,avg_loglik, mse, mae = 0, 0, 0,0
    mean_mae, mean_mse = 0, 0
    individual_nlls = []
    with torch.no_grad():
        for train_batch in train_loader:
            subsampled_mask = make_masks(train_batch, frac=frac, forecast=forecast)
            ######################
            errorbars = torch.swapaxes(train_batch[:,:,:,2], 2,1)
            weights = errorbars.clone()
            weights[weights!=0] = 1 / weights[weights!=0]
            errorbars[errorbars!=0] = torch.log(errorbars[errorbars!=0])
            logerr = errorbars.to(device)
            weights = weights.to(device)
            ######################
            train_batch = train_batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(subsampled_mask, train_batch[:,:,:,1])

            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1)
            recon_context_y = torch.cat((
              train_batch[:,:,:,1] * recon_mask, recon_mask
            ), 1).transpose(2,1)
            
            loss_info = net.compute_unsupervised_loss(
              train_batch[:, 0, :,0],
              context_y,
              train_batch[:, 0, :,0],
              recon_context_y,
              logerr,
              weights,
              num_samples=k_iwae,
              predict=True
            )
            
            individual_nlls.append(loss_info.loglik_per_ex.cpu().numpy())
            
            num_context_points = recon_mask.sum().item()
            train_loss += loss_info.composite_loss.item()* num_context_points
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n
        )
    )
    return -avg_loglik / train_n, mse / train_n, np.concatenate(individual_nlls, axis=1)[0]

def predict(dataloader, net, device='mps', subsample=False, target_x=None, forecast=False):
    k_iwae=2
    pred_mean, pred_std = [], []
    qz_mean, qz_std = [], []
    mask = []
    target = []
    tp =[]
    target_tp = []
    np.random.seed(0)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f'{i},', end='', flush=True)
            batch_len = batch.shape[0]
            if subsample == True:
                
                subsampled_mask = make_masks(batch, frac=0.5)
            else:
                subsampled_mask = make_masks(batch, frac=1.0, forecast=forecast)
                
            batch = batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(batch[:,:,:,1], subsampled_mask)
            context_y = torch.cat((batch[:,:,:,1] * subsampled_mask, subsampled_mask), 1).transpose(2,1)
            recon_context_y=torch.cat((batch[:,:,:,1] * recon_mask, recon_mask), 1).transpose(2,1)
            if target_x is not None:
                tx = torch.tensor(target_x[i*batch_len:(i*batch_len + batch_len)])[:,0]
            else:
                tx = batch[:, 0, :,0]

            px, qz = net.get_reconstruction(batch[:, 0, :,0], context_y, tx, num_samples=k_iwae,predict=True)
            pred_mean.append(px.mean.cpu().numpy())
            pred_std.append(torch.exp(0.5 * px.logvar).cpu().numpy())
            qz_mean.append(qz.mean.cpu().numpy())
            qz_std.append(torch.exp(0.5 * qz.logvar).cpu().numpy())
            target.append((batch[:, :, :,1]).cpu().numpy())
            mask.append(subsampled_mask.cpu().numpy())
            tp.append(batch[:, 0, :,0].cpu().numpy())
            target_tp.append(tx.cpu().numpy())
            
    # reconstructions
    pred_mean = np.concatenate(pred_mean, axis=1)
    pred_std = np.concatenate(pred_std, axis=1)
    # latent space 
    qz_mean = np.concatenate(qz_mean, axis=0)
    qz_std = np.concatenate(qz_std, axis=0)
    
    # targets are all ys, masks are masks, inputs are masked ys
    target = np.concatenate(target, axis=0)
    mask = np.concatenate(mask, axis=0)
    inputs = np.ma.masked_where(mask < 1., target)
    inputs = np.transpose(inputs, [0,2,1])
    target = np.transpose(target, [0,2,1])
    
    # target tp are points we'd like to project to
    tp = np.concatenate(tp, axis=0)
    target_tp = np.concatenate(target_tp, axis=0)
    pred_mean = pred_mean.mean(0)
    pred_std = pred_std.mean(0)
    recons = {'target_tp':target_tp, 'pred_mean':pred_mean,'pred_std':pred_std}
    z = {'qz_mean':qz_mean, 'qz_std':qz_std}
    examples = {'tp':tp,'target':target,'mask':mask,'inputs':inputs}
    
    return examples, z, recons


def save_recon(examples, recons, z, obj_name, bands=['r','g'], one_ex=False, save_folder='interpolations'):
    target_tp = recons['target_tp']
    pred_mean = recons['pred_mean']
    pred_std = recons['pred_std']
    
    target = examples['target']
    inputs = examples['inputs']
    tp = examples['tp']
    
    # one example
    pred_mean= pred_mean.mean(0)[np.newaxis]  
    pred_std = pred_std.mean(0)[np.newaxis]
    z_mean = z['qz_mean'].mean(0)
    z_std = z['qz_std'].mean(0)
    zs = np.concatenate((z_mean.reshape(-1,1), z_std.reshape(-1,1)),axis=1)
    
    ######### SAVING INTERPOLATIONS ###############
    pred_t = target_tp[0].nonzero()[0]
    t = target_tp[:1,pred_t]
    if os.path.isdir(save_folder):
        obj_folder = os.path.join(save_folder, obj_name)
        if not os.path.isdir(obj_folder):
            os.mkdir(obj_folder)
        for i,band in enumerate(bands):
            mean_mag = pred_mean[:,pred_t,i]
            mag_std = pred_std[:,pred_t,i]
            lc = np.concatenate((t, mean_mag, mag_std), axis=0).T
            save_file = os.path.join(save_folder,obj_name,f'{obj_name}_{band}.dat')
            
            np.savetxt(save_file, lc)
            print(f'{lc.shape=} and {band=} saved to {save_file}')
            
        z_save_file = os.path.join(save_folder,obj_name,f'{obj_name}_qz.dat')
        np.savetxt(z_save_file, zs)
        
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_normal_pdf(x, mean, logvar, mask, logerr=0.):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    logerr = logerr * mask
    logvar = logvar + logerr
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask


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

