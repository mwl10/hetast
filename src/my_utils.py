import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


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


# visualisation for one light curve w/ increasing number of points
def viz_per_example(example, net, device="cuda", k_iwae=10, fracs=[0.2,0.2,0.5]): 
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
            example = np.concatenate((example, smask[:,:,np.newaxis], rmask[:,:,np.newaxis]), axis=-1) # format the masks 

            # CUDA~
            example = torch.tensor(example)
            example = example.to(device)
            
            subsampled_mask = example[:,:,3:4]
            seqlen = example.size(0)
            # 
            context_y = torch.cat((example[:,:, 1:2] * subsampled_mask, subsampled_mask), -1)
            # probabilities per batch 
            px, _ = net.get_reconstruction(example[:,:, 0], context_y, example[:,:, 0], num_samples=k_iwae)
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
    plt.figure(figsize=(12.5, 1.5))
    for j in range(3):
        plt.subplot(1, 3, j + 1)
        plt.fill_between(tp[j], quantile1[j, :, 0], quantile2[j, :, 0], alpha=0.6, facecolor='#65c9f7', interpolate=True)
        plt.plot(tp[j], median[j, :, 0], c='b', lw=w, label='Reconstructions')
        #plt.plot(tp[n_max * j + index], gt[index], c='r', lw=w, label='Ground Truth')
        plt.scatter(tp[j], inputs[j, :, 0], c='k', lw=w, label='Observed Data')
    plt.show()

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
      
    
    return float(- avg_loglik / train_n)






def main():
    pass
    
    
if __name__ == "__main__":
    main()





