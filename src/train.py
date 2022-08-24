import numpy as np
import torch
import torch.optim as optim
import my_utils
import argparse
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
    device = torch.device(args.device)
    
    if args.data_obj: # we can make the synthetic data, or we can pass the data_obj 
        pass
    else:
        data_obj = my_utils.get_synthetic_data(seed=seed)
        
        
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"]
    union_tp = data_obj['union_tp']
    
    if args.checkpoint:
        net, optimizer, args, epoch, loss = my_utils.load_checkpoint(args.checkpoint, data_obj)
        print(f'loaded checkpoint with loss: {loss}')
    else:
        net = models.load_network(args, dim, union_tp)
        params = list(net.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        epoch = 1
        loss = 10000
        
    
#     net = models.load_network(args, dim, union_tp)
#     params = list(net.parameters())
#     optimizer = optim.Adam(params, lr=args.lr)
#     print('parameters:', utils.count_parameters(net))
    
    # stop if loss doens't decrease for how many epochs? 
    best_loss = loss 
    counter = 0
    
    for itr in range(epoch, epoch+args.niters + 1):
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
        # could add a delta requirement to this
        if args.early_stopping:
            if best_loss > (train_loss / train_n): 
                counter += 1
            else:
                best_loss = (train_loss / train_n) 
                counter == 0
            
            if counter >= args.patience:
                print(f'training has not improved for {args.patience} epochs')
                break
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8) #batch_size=8,
    parser.add_argument('--niters', type=int, default=10) # niters=10,
    parser.add_argument('--bound-variance', action='store_true') # bound_variance=False,
    parser.add_argument('--const-var', action='store_true') # const_var=False,
    parser.add_argument('--dropout', type=float, default=0.0) # dropout=0.0, 
    parser.add_argument('--elbo-weight', type=float, default=4.1) # elbo_weight=4.108914123847402,
    parser.add_argument('--embed-time', type=int, default=32) # embed_time=32, 
    parser.add_argument('--enc-num-heads', type=int, default=4) # enc_num_heads=4,
    parser.add_argument('--intensity', action='store_false') #  intensity=True,
    parser.add_argument('--k-iwae', type=int, default=1) #  k_iwae=1, 
    parser.add_argument('--kl-annealing', action='store_false') # kl_annealing=False,
    parser.add_argument('--kl-zero', action='store_true') # kl_zero=False, 
    parser.add_argument('--latent-dim', type=int, default=32) # latent_dim=32, 
    parser.add_argument('--lr', type=float, default=0.001) # lr=0.001, 
    parser.add_argument('--mixing', type=str, default='concat') # mixing='concat',
    parser.add_argument('--mse-weight', type=float, default=4.0) # mse_weight=4.060280688730988,
    parser.add_argument('--net', type=str, default='hetvae') # net='hetvae', 
    parser.add_argument('--norm', action='store_false') # norm=True,
    parser.add_argument('--normalize-input', type=str, default='znorm') # normalize_input='znorm',
    parser.add_argument('--num-ref-points', type=int, default=32) # num_ref_points=32,
    parser.add_argument('--rec-hidden', type=int, default=32) # rec_hidden=32, 
    parser.add_argument('--recon-loss', action='store_true') #  recon_loss=False,
    parser.add_argument('--sample-tp', type=float, default=0.5) # sample_tp=0.4733820088130086, 
    parser.add_argument('--save', action='store_false') # save=True, 
    parser.add_argument('--seed', type=int, default=0) # seed=0, 
    parser.add_argument('--shuffle', action='store_false') #shuffle=True,
    parser.add_argument('--std', type=float, default=0.1) # std=0.1, 
    parser.add_argument('--var-per-dim', action='store_true') # var_per_dim=False,
    parser.add_argument('--width', type=int, default=512) # width=512,
    parser.add_argument('--device', type=str, default='cuda') # device='mps'
    parser.add_argument('--data_obj', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default='50')
    
    args = parser.parse_args()
    main(args)
    