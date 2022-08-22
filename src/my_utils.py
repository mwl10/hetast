import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal 
from glob import glob
from dataset import DataSet
from eztao.carma import DRW_term, DHO_term
from eztao.ts import gpSimRand


def get_synthetic_data(num_samples=100, dims= 1, batch_size=8, sim_params={'SNR':10, 'duration':10*365, 'N':200}, drw_kernel_params={'tau':100, 'amp':0.2}, dho_kernel_params = {'a1':1.0, 'a2':1.0, 'b0':1.0,'b1':1.0}, kernel='DRW'):
##### making synthetic data multivariate
    
    if kernel=='DRW':
        log_amp, log_tau = np.log([drw_kernel_params['amp'], drw_kernel_params['tau']])
        kernel = DRW_term(log_amp, log_tau)

    elif kernel=='DHO':
        log_a1,log_a2,log_b0,log_b1 = np.log([dho_kernel_params['a1'], dho_kernel_params['a2'], dho_kernel_params['b0'], dho_kernel_params['b1']])
        kernel=DHO_term(log_a1,log_a2,log_b0,log_b1)
        
    SNR, duration, N = sim_params['SNR'], sim_params['duration'], sim_params['N']
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
        subset_lengths = np.random.randint(low=50, high=100, size=dims)
        # each dimension could be a random subset of the original, scaled, shifted,
        for j in range(dims):   
            obs_points = np.sort(np.random.choice(np.arange(num_samples), size=subset_lengths[j], replace=False))
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
            recon_mask = np.logical_xor(subsampled_mask, recon_mask)
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
    print(samples.shape)
    splindex = int(np.floor(.8*len(samples)))
    training, valid, test = np.split(samples, [splindex, splindex + int(np.floor(.1*len(samples)))])# shuffle?
    print(training.shape, valid.shape, test.shape)
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


def train(net, optimizer,epoch, train_loader, args, device="cuda", frac=0.5, errors=False, beta=0.1):
      
    train_loss = 0.
    train_n = 0.
    avg_loglik, avg_kl, mse, mae = 0., 0., 0., 0.
    for i, train_batch in enumerate(train_loader):
        batch_len = train_batch.shape[0] 
        #train_batch[:,:,2] = torch.ones((train_batch[:,:,3].shape))
        recon_mask, subsampled_mask = make_masks(train_batch, frac=0.5)
        
        train_batch = torch.cat((train_batch, torch.unsqueeze(subsampled_mask, 2), torch.unsqueeze(recon_mask, 2)), axis=-1)
        # print(torch.unsqueeze(subsampled_mask, 2).shape)
        # print(train_batch.shape)
        # train_batch[:,:,4:5] = torch.unsqueeze(recon_mask[:,:], 2)
        # train_batch[:,:,3:4] = torch.unsqueeze(subsampled_mask[:,:], 2)
        train_batch = train_batch.to(device)
        x = train_batch[:,:,0]
        y = train_batch[:,:,1:2]
        subsampled_mask = train_batch[:,:,3:4]
        recon_mask = train_batch[:,:,4:5]
        if errors:
            sample_weight = train_batch[:,:,2:3]
        else:
            sample_weight = 1.
        # weights for loss in analogy to standard weighted least squares error 

        seqlen = train_batch.size(1) 
        # subsampled flux values and their corresponding masks....
        context_y = torch.cat((
            y * subsampled_mask, subsampled_mask
        ), -1) 
        recon_context_y = torch.cat((            # flux values with only recon_mask values showing
                y * recon_mask, recon_mask
            ), -1) 
# format: compute_unsupervised_loss(self, context_x, context_y, target_x, target_y, num_samples=1, beta=1):
        loss_info = net.compute_unsupervised_loss(
            x,
            context_y,  
            x,  # can pick the points we want to project to
            recon_context_y,
            num_samples=args.k_iwae, # 1? 
            beta=beta, # beta i s a 
            # optional, will be zero if not set
            sample_weight = sample_weight,

        )
        optimizer.zero_grad()
        loss_info.composite_loss.backward()
        optimizer.step()
        #scheduler.step()
        train_loss += loss_info.composite_loss.item() * batch_len
        avg_loglik += loss_info.loglik * batch_len
        avg_kl += loss_info.kl * batch_len
        mse += loss_info.mse * batch_len
        mae += loss_info.mae * batch_len
        train_n += batch_len
        
    if epoch % 100 == 0:
        print(
            'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
            'mse: {:.6f}, mae: {:.6f}'.format(
                epoch,
                train_loss / train_n,
                -avg_loglik / train_n,
                avg_kl / train_n,
                mse / train_n,
                mae / train_n
            )
        )
        
    return -avg_loglik / train_n, mse / train_n, y, recon_mask, subsampled_mask


def make_masks(lcs, frac=0.7):
    # will depend on dimensions later
    subsampled_mask = torch.zeros_like(lcs[:, :, 1])
    recon_mask = torch.zeros_like(lcs[:, :, 1])
    for i, lc in enumerate(lcs):
        indexes = lc[:, 1].nonzero()[:,0]
        # this should vary to some extent
        length = int(np.round(len(indexes) * frac))
        obs_points = np.sort(np.random.choice(
            indexes, size=length, replace=False))
        
        subsampled_mask[i, obs_points] = 1

        recon_mask[i] = torch.logical_xor(lc[:, 1], subsampled_mask[i])
    return subsampled_mask, recon_mask



# visualisation for one light curve w/ increasing number of points
def viz_per_example(example, target_x, net, device="cuda", k_iwae=10, fracs=[0.25,0.5,0.75]): 
    pred_mean, pred_std = [], []
    masks = []
    targets = []
    tp =[]
    example = example[np.newaxis, :,:]

    np.random.seed(0)
    with torch.no_grad():
        for frac in fracs: # 
            if torch.is_tensor(example):
                example = example.cpu().numpy()

            # make new masks relative to fraction of points we got to predict w/ 
            smask, rmask = make_masks(example, frac=frac)
            target_x
            example = np.concatenate((example, smask[:,:,np.newaxis], rmask[:,:,np.newaxis]), axis=-1) # format the masks 

            
            example = torch.tensor(example)
            example = example.to(device)
            
            subsampled_mask = example[:,:,3:4]
            seqlen = example.size(0)
            # 
            context_y = torch.cat((example[:,:, 1:2] * subsampled_mask, subsampled_mask), -1)
            # probabilities per batch 
            px, qz = net.get_reconstruction(example[:,:, 0], context_y, example[:,:, 0], num_samples=k_iwae)
            print(qz.mean.shape)
            pred_mean.append(px.mean.cpu().numpy())
            # changing from logvar to std 
            pred_std.append(torch.exp(0.5 * px.logvar).cpu().numpy())

            targets.append((example[:,:, 1:2]).cpu().numpy())
            masks.append(subsampled_mask.cpu().numpy())
            tp.append(example[:,:, 0].cpu().numpy())

    pred_mean = np.concatenate(pred_mean, axis=1)
    pred_std = np.concatenate(pred_std, axis=1)
    targets = np.concatenate(targets, axis=0)
    masks = np.concatenate(masks, axis=0)
    tp = np.concatenate(tp, axis=0)
    inputs = np.ma.masked_where(masks < 1., targets)
    print(f'pred_mean: {pred_mean.shape}', f'pred_std: {pred_std.shape}=', f'targets: {targets.shape}', f'masks: {masks.shape}', f'tps: {tp.shape}', f'inputs: {inputs.shape}')
    # we're are sampling from the intermediate representation w/ k_iwae
    # then we sample from the means/stds from the intermediate representation w/ k_iwae
    
    preds = np.random.randn(k_iwae // 2, k_iwae, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3]) * pred_std + pred_mean
    preds = preds.reshape(-1, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3])
    print(preds.shape)
    median = preds.mean(0)
    quantile2 = np.quantile(preds, 0.859, axis=0)
    quantile1 = np.quantile(preds, 0.141, axis=0)
    
    w = 2.0
    plt.figure(figsize=(25, 3))
    for j in range(1):
        plt.subplot(1, 3, j + 1)
        plt.fill_between(tp[j], quantile1[j, :, 0], quantile2[j, :, 0], alpha=0.6, facecolor='#65c9f7', interpolate=True)
        plt.plot(tp[j], median[j, :, 0], c='b', lw=w, label='Reconstructions')
        #plt.plot(tp[n_max * j + index], gt[index], c='r', lw=w, label='Ground Truth')
        plt.scatter(tp[j], inputs[j, :, 0], c='k', lw=w, label='Observed Data')
    plt.show()
    return qz


def evaluate_hetvae(
    net,
    dim,
    train_loader,
    sample_tp=0.5,
    shuffle=False,
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
            subsampled_mask = train_batch[:,:,:,4]
            recon_mask = train_batch[:,:,:,5]
            
            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1) # torch.Size([128, 203, 82]) # batch size, length, dims
            # torch.Size([128, 203])
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



def main():
    pass
    
    
if __name__ == "__main__":
    main()





