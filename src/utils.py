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
import logging
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand, gpSimFull


# sys.path.insert(0, '/Users/mattlowery/Desktop/code/astro/hetvae/src/reverberation_mapping')
# import photRM

def save_synth_data(base_folder='/Users/mattlowery/Desktop/code/astro/hetvae/src/test_data', save_folder='synth_test_data', seed = 0, kernel='drw',duration=730, n=180, uniform=False):
    """
    This function creates and loads a synthetic dataset relative to a given kernel (drw or dho), 
    distributing the kernel params relative to a real dataset  
    
    parameters:
        folder             (str)      --> synthetic data will be fit to the light curves in this folder to get an honest distribution of params
         ----- optional -----
        seed               (int)      --> random seed, this matters to keep the shuffles consistent
        batch_size         (int)      --> for the network, usually a multiple of 2
        kernel             (str)      --> 'drw' (dampled random walk), a carma(1) process; 'dho' (damped harmonic oscillator), a carma(2,1) process
        n
        duration 
        
    drw_kernel params --> 'tau' is decorelation timescale, 'amp' is the amplitude
    dho_kernel params --> a1 = 2 * Xi * w0
                       a2 = w0 ** 2
                       b0 = sigma
                       b1 = tau * b0
                       wherein Xi is the damping ratio, w0 is the natural oscillation frequency, sigma is the amplitude
                       of the short term perturbing white noise, tau is the characteristic timescale of the perturbation process
    returns:
        a dictionary of torch dataloaders with data formatted as necessary for network training 
        , as well as the dimension and union of all the time points
    """
    synth_band_name = 'simband1'
    
    np.random.seed(seed)
    if not os.path.isdir(base_folder):
        raise Exception(f"{base_folder} is not a directory")
        
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
        os.mkdir(os.path.join(save_folder,synth_band_name))   
        
    lcs = DataSet(name=base_folder, min_length=50, sep=',', start_col=1)
    band_folders = os.listdir(base_folder)
    for band_folder in band_folders:
        band = band_folder.lower()
        lcs.add_band(band, os.path.join(base_folder, band_folder))
        
    lcs.filter()         
    lcs.prune_outliers()
    lcs.set_carma_fits(kernel=kernel)
    lcs.set_snr()
    
    synth_lcs = []
    for i, params in enumerate(lcs.carma_fits):
        if np.isnan(params[0]):
            continue
        if kernel == 'drw':
            term = DRW_term(*np.log(params))
        else:
            term = DHO_term(*np.log(params))
                
        # kernel, snr, duration (days), n      
        if uniform == True:
            lc = np.array(gpSimFull(term,lcs.snr[i,0], duration, n)).transpose(1,0)
        else: 
            lc = np.array(gpSimRand(term,lcs.snr[i,0], duration, n)).transpose(1,0)   
        fpath = f'{os.path.join(save_folder, synth_band_name,str(i))}_.csv'
        np.savetxt(fpath,lc, delimiter=',', header=' , , ', comments='')



def get_data(folder, seed= 0, sep=',', start_col=0, batch_size=8, min_length=40, n_union_tp=1000, num_resamples=0):
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
    
    np.random.seed(seed) # for the shuffle
    if not os.path.isdir(folder):
        raise Exception(f"{folder} is not a directory")
    ##################################
    # initializing the DataSet objects 
    ##################################
    lcs = DataSet(name=folder, min_length=min_length, sep=sep, start_col=start_col)
    band_folders = os.listdir(folder)
    for band_folder in band_folders:
        band = band_folder.lower()
        lcs.add_band(band, os.path.join(folder, band_folder))
    ### preprocessing functions ####################################################################
    lcs.filter()
    lcs.prune_graham()
    lcs.chop_lcs()
    lcs.resample_lcs(num_resamples=num_resamples)
    ###################################
    lcs.set_excess_vars()
    lcs.set_mean_mags()
    ###################################
    lcs.normalize() 
    lcs.formatting()
    lcs.set_union_tp(uniform=True,n=n_union_tp) #maybe do this as some regularly sequenced bit
    print(f'dataset created w/ shape {lcs.dataset.shape}')
    ######## done preprocessing ########################################################################################################
    lcs.set_data_obj(batch_size=batch_size)
    return lcs


def preview_masks(dataloader, bands, batch_num = 0, frac=0.5, N=1, figsize=(15,15)):
    """
    function to visualize masks on the light curves
    
    parameters:
        dataloader
        bands      (list)   --> list of bands for labeling the plots
        
        ----- optional -----
        batch_num  (int)    0        --> which batch you want to plot
        frac       (float)  0.5      --> fraction of points to mask 
        N          (int)    1        --> number of object's light curves to plot
        figsize    (tuple)  (15,15)  --> size of the plotting
    """
    batch_num = batch_num % len(dataloader) # in case batch_num given is out of range
    for i, batch in enumerate(dataloader):
        if batch_num == i:
            dims = batch.size(1)
            subsampled_mask = make_masks(batch, frac=frac)
            fig, ax = plt.subplots(N, dims, figsize=figsize, squeeze=False)
            if N > len(batch): N = len(batch)  # in case num light curves is out of range
            print(N)
            for j in range(N):
                for k in range(dims):
                    t = batch[j,k,:,0]
                    y = batch[j,k,:,1]
                    sub = subsampled_mask[j,k,:]
                    rec = torch.logical_xor(sub, y)
                    ## ignoring zeros as masks or as padding
                    sub = (y*sub).nonzero()[:,0]
                    rec = (y*rec).nonzero()[:,0]
                    ax[j][k].scatter(t[rec],y[rec], c='black', marker='x', zorder=30, label='masked', s=100)
                    ax[j][k].scatter(t[sub],y[sub], c='blue', label='input')
                    
    lines_labels = ax[0][0].get_legend_handles_labels()
    lines,labels = lines_labels[0], lines_labels[1]
    fig.legend(lines, labels, bbox_to_anchor=(0.12, 0.92), loc='upper left')
    [ax[j][index].set_xlabel(bands[index]) for index in range(len(bands))]

    
def preview_lcs(dataloader, bands, batch_num = 0, N=1, figsize=(15,15)):
    """
    preview some of the loaded light curves
    
    parameters:
        dataloader
        bands      (list)   --> list of bands for labeling the plots
        
        ----- optional -----
        batch_num  (int)    0        --> which batch you want to plot 
        N          (int)    1        --> number of object's light curves to plot
        figsize    (tuple)  (15,15)  --> size of the plotting
    """
    batch_num = batch_num % len(dataloader) # in case batch_num given is out of range
    for i, batch in enumerate(dataloader):
        if batch_num == i:
            dims = batch.size(1)
            fig, ax = plt.subplots(N, dims, figsize=figsize, squeeze=False)
            if N > len(batch): N = len(batch)  # in case num light curves is out of range
            for j in range(N):
                for k in range(dims):
                    t = batch[j,k,:,0]
                    y = batch[j,k,:,1]
                    yerr = batch[j,k,:,2]
                    ## ignoring zeros as masks or as padding
                    pts = y.nonzero()[:,0]
                    ax[j][k].errorbar(t[pts],y[pts], yerr=yerr[pts], c='blue', fmt='.', markersize=4, ecolor='red', elinewidth=1, capsize=2)
                    
    lines_labels = ax[0][0].get_legend_handles_labels()
    lines,labels = lines_labels[0], lines_labels[1]
    fig.legend(lines, labels, bbox_to_anchor=(0.12, 0.92), loc='upper left')
    [ax[j][index].set_xlabel(bands[index]) for index in range(len(bands))]
     
def load_checkpoint(filename, data_obj):
    """
    loads a model checkpoint 
    """
    if os.path.isfile(filename):
        logging.info("=> loading checkpoint '{}'".format(filename))
        cp = torch.load(filename)
        logging.info(cp['args'])
        net = model.load_network(cp['args'], data_obj['input_dim'], data_obj['union_tp'])
        net.load_state_dict(cp['state_dict'])
        params = list(net.parameters())
        optimizer = optim.Adam(params)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        return net,optimizer, cp['args'], cp['epoch'], cp['loss']
    

def make_masks(batch, frac=0.5):
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
    sample_tp=0.5,
    k_iwae=1,
    device='cuda',
):
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)
    train_n = 0
    train_loss,avg_loglik, mse, mae = 0, 0, 0,0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            subsampled_mask = make_masks(train_batch)
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
            )
            num_context_points = recon_mask.sum().item()
            train_loss += loss_info.composite_loss.item()* num_context_points
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    logging.info(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n
        )
    )
    #return train_loss / train_n
    return -avg_loglik / train_n, mse / train_n,

def predict(dataloader, net, device='mps', k_iwae=2, subsample=False, target_x=None, plot=True, figsize=(25,15)):
    pred_mean, pred_std = [], []
    qz_mean, qz_std = [], []
    masks = []
    targets = []
    tp =[]
    target_tp = []
    np.random.seed(0)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_len = batch.shape[0]
            
            if subsample == True:
                subsampled_mask = make_masks(batch, frac=0.5)
            else:
                subsampled_mask = make_masks(batch, frac=1.0)
                
            batch = batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(batch[:,:,:,1], subsampled_mask)
            context_y = torch.cat((batch[:,:,:,1] * subsampled_mask, subsampled_mask), 1).transpose(2,1)
            recon_context_y=torch.cat((batch[:,:,:,1] * recon_mask, recon_mask), 1).transpose(2,1)
            if target_x is not None:
                tx = torch.tensor(target_x[i*batch_len:(i*batch_len + batch_len)])[:,0]
            else:
                tx = batch[:, 0, :,0]

            px, qz = net.get_reconstruction(batch[:, 0, :,0], context_y, tx, num_samples=k_iwae)
            pred_mean.append(px.mean.cpu().numpy())
            pred_std.append(torch.exp(0.5 * px.logvar).cpu().numpy())
            qz_mean.append(qz.mean.cpu().numpy())
            qz_std.append(torch.exp(0.5 * qz.logvar).cpu().numpy())
            
            targets.append((batch[:, :, :,1]).cpu().numpy())
            masks.append(subsampled_mask.cpu().numpy())
            tp.append(batch[:, 0, :,0].cpu().numpy())
            target_tp.append(tx.cpu().numpy())
            
      
    pred_mean = np.concatenate(pred_mean, axis=1)
    pred_std = np.concatenate(pred_std, axis=1)
    qz_mean = np.concatenate(qz_mean, axis=0)
    qz_std = np.concatenate(qz_std, axis=0)
    
    targets = np.concatenate(targets, axis=0)
    masks = np.concatenate(masks, axis=0)
    tp = np.concatenate(tp, axis=0)
    target_tp = np.concatenate(target_tp, axis=0)
    inputs = np.ma.masked_where(masks < 1., targets)
    targets = np.transpose(targets, [0,2,1])
    inputs = np.transpose(inputs, [0,2,1])
    # reparam trick
    preds = np.random.randn(k_iwae//2, k_iwae, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3]) * pred_std + pred_mean
    preds = preds.reshape(-1, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3])
    
    qz_preds = np.random.randn(k_iwae//2, k_iwae, qz_mean.shape[0], qz_mean.shape[1], qz_mean.shape[2]) * qz_std + qz_mean
    qz_preds = qz_preds.reshape(-1, qz_mean.shape[0], qz_mean.shape[1], qz_mean.shape[2])
    qz_median = preds.mean(0)
    if plot == True:
        median = preds.mean(0)
        quantile2 = np.quantile(preds, 0.841, axis=0)
        quantile1 = np.quantile(preds, 0.159, axis=0)
        fig,ax = plt.subplots(5,1,figsize=figsize)
        for i in range(4):
            #### remove padding and masked vals
            nonzero_pred = target_tp[i].nonzero()[0]
            nonzero_in = inputs[i].nonzero()[0]
            nonzero_targets = targets[i].nonzero()[0]
            
            ax[i].fill_between(target_tp[i,nonzero_pred], quantile1[i,nonzero_pred,0], quantile2[i,nonzero_pred,0], label='error envelope',color='lightcoral')
            ax[i].errorbar(target_tp[i, nonzero_pred], median[i, nonzero_pred,0], yerr=(quantile2[i,nonzero_pred,0] - median[i, nonzero_pred,0]), c='blue', ecolor='#65c9f7', label='prediction')
            # conditoned on points
            #ax[i].scatter(tp[i, nonzero_in], inputs[i, nonzero_in], c='black', marker='x', zorder=30, label='conditioned on', s=100)
            # all points 
            ax[i].scatter(tp[i, nonzero_targets], targets[i, nonzero_targets], c='green', marker='.', s=100, zorder=100)       
    return qz_preds, tp, targets, inputs, target_tp, preds


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

