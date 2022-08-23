import argparse
import numpy as np
import torch
import torch.optim as optim
import my_utils
from argparse import Namespace

from random import SystemRandom
import models
import utils


def main(args):
    experiment_id = int(SystemRandom().random() * 10000000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('mps' if torch.has_mps else 'cpu')
    
    #if args.dataset == 'synth':
    data_obj = my_utils.get_synthetic_data(seed=seed)

    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    union_tp = data_obj['union_tp']
    net = models.load_network(args, dim, union_tp)

    
    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(net))
    
    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
        if args.kl_annealing:
            wait_until_kl_inc = 10000
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.999999 ** (itr - wait_until_kl_inc))
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
          
            subsampled_mask = train_batch[:,:,:,4]
            recon_mask = train_batch[:,:,:,5]

            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask
            ), 1).transpose(2,1)
        
            recon_context_y = torch.cat((
                train_batch[:,:,:,1] * recon_mask, recon_mask
            ), 1).transpose(2,1)
            
            #print(recon_context_y.shape, context_y.shape,train_batch[:, 0, :,0].shape )
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, 0, :,0],
                context_y,
                train_batch[:, 0, :,0],
                recon_context_y,
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            
        print(itr)
        if itr % 10 == 0:
          print(
              'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
              'mse: {:.6f}, mae: {:.6f}'.format(
                  itr,
                  train_loss / train_n,
                  -avg_loglik / train_n,
                  avg_kl / train_n,
                  mse / train_n,
                  mae / train_n
              )
          )
        if itr % 10 == 0:
            for loader, num_samples in [(val_loader, 5), (test_loader, 100)]:
                my_utils.evaluate_hetvae(
                    net,
                    dim,
                    loader,
                    0.5,
                    shuffle=False,
                    k_iwae=num_samples,
                    device='mps'
                )
        ## set a learning rate scheduler?
        ## set a stopping for num_iters where there's no improvement
        if itr % 10 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / train_n,
            }, 'synth' + '_' + str(experiment_id) + '.h5')



if __name__ == '__main__':
    
    args = Namespace(batch_size=8,
                    niters=10,
                    bound_variance=False, 
                    const_var=False,
                    dropout=0.0, 
                    elbo_weight=4.108914123847402, 
                    embed_time=32,           
                    enc_num_heads=4, 
                    intensity=True, 
                    k_iwae=1, 
                    kl_annealing=False, 
                    kl_zero=False, 
                    latent_dim=32, 
                    lr=0.001, 
                    mixing='concat', 
                    mse_weight=4.060280688730988, 
                    net='hetvae', 
                    norm=True, 
                    normalize_input='znorm', 
                    num_ref_points=32, 
                    rec_hidden=32, 
                    recon_loss=False, 
                    sample_tp=0.4733820088130086, 
                    save=True, 
                    seed=0, 
                    shuffle=True, 
                    std=0.1, 
                    var_per_dim=False, 
                    width=512, 
                    device='mps')

    main(args)
    