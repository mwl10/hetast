import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from dataset import DataSet
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand
import os
import model
import torch.optim as optim
import logging


def get_data(folder, seed= 0, sep=',', start_col=0, batch_size=8, min_length=10):
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
        
    lcs.preprocess()
    lcs.set_data_obj(batch_size=batch_size)
    
    logging.info(f'created dataset of shape {lcs.dataset.shape}')
    return lcs


def preview_mask(dataloader, bands, batch_num = 0, frac=0.5, N=1, figsize=(15,15)):
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
            if target_x is not None:
                tx = torch.tensor(target_x[i*batch_len:(i*batch_len + batch_len)])[:,0]
            else:
                tx = batch[:, 0, :,0]
            print(tx.shape)
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
        for i in range(1):
            #### remove padding and masked vals
            nonzero_pred = target_tp[i].nonzero()[0]
            nonzero_in = inputs[i].nonzero()[0]
            
            ax[i].fill_between(target_tp[i,nonzero_pred], quantile1[i,nonzero_pred,0], quantile2[i,nonzero_pred,0], alpha=0.6, facecolor='#65c9f7', interpolate=True)
            ax[i].plot(target_tp[i, nonzero_pred], median[i, nonzero_pred])
            ax[i].scatter(tp[i, nonzero_in], inputs[i, nonzero_in])
        
    return preds, qz_preds, tp, target_tp, inputs 

