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
    
    if args.data_folder == 'synth':
        data_obj = my_utils.get_synthetic_data(seed=seed, uniform=True)
    else:
        _, data_obj = my_utils.get_data(seed = seed, folder=args.data_folder, start_col=args.start_col)

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
        
    model_size = utils.count_parameters(net) 
    
    if args.scheduler == True: 
        args.lr = my_utils.update_lr(model_size, epoch, args.warmup)
        for g in optimizer.param_groups:
            g['lr'] = args.lr
            
    
    # stop if loss doesn't decrease for how many epochs? 
    best_loss = loss 
    counter = 0
    
    for itr in range(epoch, epoch+args.niters):
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
            
        if args.scheduler == True: 
            args.lr = my_utils.update_lr(model_size, itr, args.warmup)
            for g in optimizer.param_groups:
                g['lr'] = args.lr
                
        print(itr, f'lr: {args.lr}')
        
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
            valid_nll_loss = my_utils.evaluate_hetvae(
                net,
                dim,
                val_loader,
                0.5,
                k_iwae=args.k_iwae,
                device=args.device
                )
        if itr % args.save_at == 0 and args.save:
            print('saving.................')
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / train_n,
            }, 'synth' + '_' + str(experiment_id) + '.h5')
            print('done')
            
        # could add a delta improvement requirement to this
        if args.early_stopping:
            if best_loss > (train_loss / train_n): 
                counter += 1
            else:
                best_loss = (train_loss / train_n) 
                counter == 0
            
            if counter >= args.patience:
                print(f'training has not improved for {args.patience} epochs')
                torch.save({
                    'args': args,
                    'epoch': itr,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss / train_n,
                }, 'synth' + '_' + str(experiment_id) + '.h5')
                break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8) 
    parser.add_argument('--niters', type=int, default=10) 
    parser.add_argument('--bound-variance', action='store_false') 
    parser.add_argument('--const-var', action='store_true') 
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--elbo-weight', type=float, default=4.1)
    parser.add_argument('--embed-time', type=int, default=32)  
    parser.add_argument('--enc-num-heads', type=int, default=4) 
    parser.add_argument('--intensity', action='store_false') 
    parser.add_argument('--k-iwae', type=int, default=1)  
    parser.add_argument('--kl-annealing', action='store_false')
    parser.add_argument('--kl-zero', action='store_true') 
    parser.add_argument('--latent-dim', type=int, default=32)  
    parser.add_argument('--lr', type=float, default=0.001)  
    parser.add_argument('--mixing', type=str, default='concat') 
    parser.add_argument('--mse-weight', type=float, default=4.0) 
    parser.add_argument('--net', type=str, default='hetvae') 
    parser.add_argument('--norm', action='store_false') 
    parser.add_argument('--normalize-input', type=str, default='znorm') 
    parser.add_argument('--num-ref-points', type=int, default=32) 
    parser.add_argument('--rec-hidden', type=int, default=32) 
    parser.add_argument('--sample-tp', type=float, default=0.5) 
    parser.add_argument('--save', action='store_false') 
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shuffle', action='store_false') 
    parser.add_argument('--std', type=float, default=0.1)  
    parser.add_argument('--var-per-dim', action='store_true')
    parser.add_argument('--width', type=int, default=512) 
    parser.add_argument('--device', type=str, default='cuda') 
    parser.add_argument('--data_obj', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--patience', type=int, default='50')
    parser.add_argument('--save-at', type=int, default='50')
    parser.add_argument('--scheduler', action='store_false')
    parser.add_argument('--warmup', type=int, default='4000')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--start-col', type=int, default='0')
    
    args = parser.parse_args()
    main(args)
    