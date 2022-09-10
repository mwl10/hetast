import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from dataset import DataSet
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand
import os
import models
import torch.optim as optim


def update_lr(model_size, itr, warmup):
    lr = (model_size ** -0.5) * min(itr**-0.5, itr * warmup**-0.5)
    return lr


# plot examples from a data loader
def plot(dataloader, n=6, figsize=(15,15)):
    batch = next(iter(dataloader))
    fig, ax = plt.subplots(n,1, figsize=figsize)
    for i in range(n):
        t = batch[i,0,:,0]
        y = batch[i,0,:,1]
        sub = batch[i,0,:,-2]
        rec = batch[i,0,:,-1]
        tsub = t*sub
        trec = t*rec
        ysub = y*sub
        yrec = y*rec
        ax[i].scatter(tsub,ysub, c='black', marker='x', zorder=30, label='input', s=100)
        ax[i].scatter(trec,yrec, c='blue', label='masked values for prediction')

    lines_labels = ax[0].get_legend_handles_labels()
    lines,labels = lines_labels[0], lines_labels[1]
    fig.legend(lines, labels, bbox_to_anchor=(0.12, 0.92), loc='upper left')
    

# checkpoint needs to load the model with the same data, union_tp, dims, if we add a scheduler it needs to be added as well 
def load_checkpoint(filename, data_obj):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        #cp keys ['args', 'epoch', 'state_dict', 'optimizer_state_dict', 'loss']
        cp = torch.load(filename)
        print(cp['args'])
        print(data_obj['input_dim'])
        net = models.load_network(cp['args'], data_obj['input_dim'], data_obj['union_tp'])
        net.load_state_dict(cp['state_dict'])
        params = list(net.parameters())
        optimizer = optim.Adam(params)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        return net,optimizer, cp['args'], cp['epoch'], cp['loss']
    

    # u = 320.5 - 393.5, g = 401.5 - 551.9, r = 552.0 - 691.0, i = 691.0 - 818.0,
    # z = 818.0 - 923.5, y = 923.8 - 1084.5


# disc transfer function, psi(tau, lambda)
def psi(mass = 2e8, l = 0.1, n=0.1, disc_inclination=45, azimuth=0):
    # convolute, shift 
    return None   


def get_data(seed=0, folder='./ZTF_DR_data/i_band', band = 'i', start_col=1, batch_size=8, min_length=10, max_length=10):
    if os.path.isdir(folder):
        files = [os.path.join(folder, file) for file in os.listdir(folder)]
    else:
        print('the folder provided is not a directory')
        return
    
    np.random.seed(seed) # for the shuffle
    
    lcs = DataSet() \
        .add_files(files) \
        .files_to_numpy(minimum=min_length, start_col=start_col, sep=',') \
        .prune_outliers() \
        .resample_dataset(num_samples=0) \
        .normalize() \
        .reorder() \
        .set_union_x() \
        .zero_fill() \
        .make_masks() 
    
    lcs.dataset = np.concatenate((lcs.dataset, lcs.subsampled_mask[:,:,np.newaxis], lcs.recon_mask[:,:,np.newaxis]), axis=-1) 
    
    shuffle = np.random.permutation(len(lcs.dataset)) # have a consistant shuffle for later
    lcs.dataset = lcs.dataset[shuffle]
    lcs.unnormalized_data = [lcs.unnormalized_data[i] for i in shuffle] # 
    
    dataset = lcs.dataset[:, np.newaxis, :, :]
    print(f'dataset shape {dataset.shape}')
    splindex = int(np.floor(.8*len(dataset)))
    training, valid, test = np.split(dataset, [splindex, splindex + int(np.floor(.1*len(dataset)))])
    # unsqueeze a dimension?
                      
    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    union_tp = torch.Tensor(lcs.union_x)
                      
    data_obj = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
        'union_tp': union_tp,
        "input_dim": 1,
    }
    return lcs, data_obj

# uniform only really means we don't subsample the other dimensions? 
# firslty it should mean that the sim data we created is uniform  
def get_synthetic_data(seed = 0, num_samples=100, uniform=False, dims= 1, batch_size=8, sim_params={'SNR':10, 'duration':10*365, 'N':200}, drw_kernel_params={'tau':100, 'amp':0.2}, dho_kernel_params = {'a1':1.0, 'a2':1.0, 'b0':1.0,'b1':1.0}, kernel='DRW'):
##### making synthetic data multivariate
    np.random.seed(seed)
    
    if kernel=='DRW':
        log_amp, log_tau = np.log([drw_kernel_params['amp'], drw_kernel_params['tau']])
        kernel = DRW_term(log_amp, log_tau)

    elif kernel=='DHO':
        log_a1,log_a2,log_b0,log_b1 = np.log([dho_kernel_params['a1'], dho_kernel_params['a2'], dho_kernel_params['b0'], dho_kernel_params['b1']])
        kernel=DHO_term(log_a1,log_a2,log_b0,log_b1)
        
    SNR, duration, N = sim_params['SNR'], sim_params['duration'], sim_params['N']
    # this is gpSimUniform if we have uniform
    synth = np.array([gpSimRand(kernel, SNR, duration, N) for _ in range(num_samples)]).transpose(0,2,1)   
    size = int(num_samples / dims)
    num_examples = synth.shape[0]
    union_tps_all = []
    data = np.zeros((num_examples, num_samples, dims))
    samples = []
    max_union_tps = []
    for i in range(num_examples):
        union_tps = []
        sample = []
        if uniform == True:
            subset_lengths = [N] * 4
        else:
            subset_lengths = np.random.randint(low=50, high=100, size=dims)
        # each dimension could be a random subset of the original, scaled, shifted,
        for j in range(dims):   
            obs_points = np.sort(np.random.choice(np.arange(N), size=subset_lengths[j], replace=False))
            sample_j = synth[i, obs_points, :]

            ##### here we could vary y vals, t vals for each dim
            #sample_j[:,0] += np.random.uniform(50,100)# shift by arbitrary amount!
            #sample_j[:,1] *= np.random.rand() # scale by arbitrary amount!

            union_tps.extend(sample_j[:, 0])
            t,y,yerr = sample_j[:,0], sample_j[:,1], sample_j[:,2]
            sample.append(sample_j)

        samples.append(sample)
        union_tps = list(set(union_tps))
        union_tps.sort() # save these 
        union_tps_all.append(union_tps)
        if len(union_tps) > len(max_union_tps):
            max_union_tps = union_tps


    for i in range(num_examples):
        sample = samples[i]
        for j in range(dims):
            sample_j = sample[j]
            new_t = union_tps_all[i]
            new_y = np.zeros(len(new_t))
            new_yerr = np.zeros(len(new_t))
            # get relative indexes
            mask = np.isin(new_t, sample_j[:,0])
            indexes = np.where(mask)[0]
            # subsampled the mask! # maybe astromer one? -------------- 
            subsampled_mask = np.zeros_like(new_y)
            recon_mask = np.zeros_like(new_y)

            length = int((mask * 1).sum() * np.random.uniform(0.35, 0.65))
            obs_points = np.sort(np.random.choice(indexes, size=length, replace=False))
            subsampled_mask[obs_points] = 1
            recon_mask = np.logical_xor(subsampled_mask, sample_j[:,1])
            #-----------------------------------------------------------
            new_y[indexes] = sample_j[:,1]
            new_yerr[indexes] = sample_j[:,2]

            need_to_append = len(max_union_tps) - len(new_t)
            sample_j = np.array([new_t, new_y, new_yerr, mask, subsampled_mask, recon_mask]).T
            sample_j = np.append(sample_j, np.zeros((need_to_append, 6)), axis=0)
            # mask subsampled masks?
            samples[i][j] = sample_j
            


    samples = np.array(samples)
    # torch loader
    samples = samples.astype('float32')
    np.random.shuffle(samples)
    #print(samples.shape)
    splindex = int(np.floor(.8*len(samples)))
    training, valid, test = np.split(samples, [splindex, splindex + int(np.floor(.1*len(samples)))])# shuffle?
    #print(training.shape, valid.shape, test.shape)
    # normalizing?
    
    train_loader = torch.utils.data.DataLoader(training, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    max_union_tps = torch.Tensor(max_union_tps)
    
    data_objects = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
        'union_tp': max_union_tps,
        "input_dim": dims,
    }
    
    return data_objects




def make_masks(lcs, frac=0.7):
    # will depend on dimensions later
    subsampled_mask = np.zeros_like(lcs[:, :, 1])
    recon_mask = np.zeros_like(lcs[:, :, 1])
    for i, lc in enumerate(lcs):
        indexes = lc[:, 1].nonzero()[0]
        # this should vary to some extent
        length = int(np.round(len(indexes) * frac))
        obs_points = np.sort(np.random.choice(
            indexes, size=length, replace=False))
        subsampled_mask[i, obs_points] = 1
        recon_mask[i] = np.logical_xor(lc[:, 1], subsampled_mask[i])
    return subsampled_mask, recon_mask


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
    avg_loglik, mse, mae = 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            subsampled_mask = train_batch[:,:,:,-2]
            recon_mask = train_batch[:,:,:,-1]
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
              num_samples=k_iwae,
            )

            num_context_points = recon_mask.sum().item()
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
    return -avg_loglik / train_n



def main():
    pass
    
    
if __name__ == "__main__":
    main()





