import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal 
from glob import glob
from dataset import DataSet

def train(net, optimizer,epoch, train_loader, args, device="cuda", frac=0.5, errors=False):
      
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
            beta=0.1,
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
    return -avg_loglik / train_n, mse / train_n

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




def evaluate(net, data_loader, device="cuda"):
    train_loss = 0.
    train_n = 0.
    avg_loglik, mse, mae = 0,0,0
    mean_mae, mean_mse = 0,0
    with torch.no_grad():
        for i, train_batch in enumerate(data_loader):
            batch_len = train_batch.shape[0] 
            train_batch = train_batch.to(device)
            x = train_batch[:,:,0]
            y = train_batch[:,:,1:2]
            subsampled_mask = train_batch[:,:,3:4]
            recon_mask = train_batch[:,:,4:]
            sample_weight = train_batch[:,:,2:3]
            seqlen = train_batch.size(1) 
            # subsampled flux values and their corresponding masks....
            context_y = torch.cat((
                y * subsampled_mask, subsampled_mask
            ), -1) 
            recon_context_y = torch.cat((            # flux values with only recon_mask values showing
                    y * recon_mask, recon_mask
                ), -1) 


      # #   def compute_unsupervised_loss(self, context_x, context_y, target_x, target_y, num_samples=1, beta=1):
            loss_info = net.compute_unsupervised_loss(
                x, # context_x, times
                context_y,    # context_y
                x, # target_x
                recon_context_y,
                num_samples=1, #???
                beta=1,
                # optional, will be one if not set
                # sample_weight=sample_weight

            )
            num_context_points = recon_mask.sum().item()
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    # print(
    # 'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
    # 'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
    #     - avg_loglik / train_n,
    #     mse / train_n,
    #     mae / train_n,
    #     mean_mse / train_n,
    #     mean_mae / train_n
    # ))
      

    return - avg_loglik / train_n




def main():
    pass
    
    
if __name__ == "__main__":
    main()





