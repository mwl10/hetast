import numpy as np
import torch
import torch.optim as optim
import argparse
from random import SystemRandom
from model import load_network
import utils
import warnings
import time
from torch.optim import lr_scheduler
import hydra
from omegaconf import DictConfig,OmegaConf
import sys
import pickle 
import os
import argparse


def train(args):
    
    experiment_id = int(SystemRandom().random() * 10000)
    print(args, experiment_id)
    
    ##################################
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    ##################################
    device = torch.device(args.device)
    if args.data_folder.split('.')[-1] == 'pkl':
        lcs = utils.load_obj(args.data_folder)
    else:
        lcs = utils.get_data(folder=args.data_folder, start_col=args.start_col, 
                             n_union_tp=args.n_union_tp, num_resamples=args.num_resamples,
                             batch_size=args.batch_size)
    
    data_obj = lcs.data_obj
    train_loader = data_obj["train_loader"]
    test_loader = data_obj["test_loader"]
    val_loader = data_obj["valid_loader"]
    dim = data_obj["input_dim"] 
    union_tp = data_obj['union_tp']
    
    if args.checkpoint:
        net,optimizer,scheduler,lrs, _,epoch, losses = \
        utils.load_checkpoint(args.checkpoint, data_obj, device=args.device)
        train_losses = losses[0]
        val_losses = losses[1]
        test_losses = losses[2]
        epoch+=1
#         for g in optimizer.param_groups:
#                 ## update learning rate for checkpoint 
#                 g['lr'] = args.lr    
        print(f'loaded checkpoint w/ {loss=}')
    
    else:
        net = load_network(args, dim, union_tp)
        params = list(net.parameters())
        optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1,args.beta2))
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.factor,
                                                   threshold=args.threshold, 
                                                   patience=args.lr_patience) #factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel',  eps=1e-08,)
        epoch = 1
        loss = 1000000
        train_losses = []
        test_losses = []
        val_losses = []
        lrs = []
        
    model_size = utils.count_parameters(net) 
    print(f'{model_size=}')
    ############### have patience ##########
    best_loss = loss
    patience_counter = 0
    ######################################## 
    if args.kl_annealing:
        kl_coefs = utils.frange_cycle_linear(6000, n_cycle=16)
    ##################
    for itr in range(epoch, epoch+args.niters):
        print(f'{itr},', end='', flush=True)
        train_loss = 0
        train_n = 0
        avg_loglik, avg_wloglik, avg_kl, mse, wmse, mae = 0, 0, 0, 0, 0, 0  
        if args.kl_annealing:
            kl_coef = kl_coefs[itr] # need global number of epochs to continue on based on schedule
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        ###################################################################  
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            ### including obs error ##############################
            errorbars = torch.swapaxes(train_batch[:,:,:,2], 2,1)
            weights = errorbars.clone()
            weights[weights!=0] = 1 / weights[weights!=0]
            errorbars[errorbars!=0] = torch.log(errorbars[errorbars!=0]**2)
            logerr = errorbars.to(device) # log variance on observations 
            weights = weights.to(device)

            ############################################################
            subsampled_mask = utils.make_masks(train_batch, frac=args.frac)
            train_batch = train_batch.to(device)
            subsampled_mask = subsampled_mask.to(device)
            recon_mask = torch.logical_xor(subsampled_mask, train_batch[:,:,:,1])
            context_y = torch.cat((
              train_batch[:,:,:,1] * subsampled_mask, subsampled_mask), 1).transpose(2,1)
            recon_context_y = torch.cat((
                train_batch[:,:,:,1] * recon_mask, recon_mask), 1).transpose(2,1)
            #############################################################
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, 0, :,0],
                context_y,
                train_batch[:, 0, :,0],
                recon_context_y,
                logerr,
                weights,
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()

            if args.inc_errors:
                loss_info.weighted_comp_loss.backward()
            else:
                loss_info.composite_loss.backward()

            optimizer.step()

            ########### train loss ##################################
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_wloglik += loss_info.wloglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            wmse += loss_info.wmse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
            #########################################################
#         print(train_loss, avg_loglik, avg_wloglik, avg_kl, mse, wmse)
        ####### nan, stop training #############
        if np.isnan(train_loss / train_n):
            print('nan in loss,,,,,,,,, stopping')
            break

        #### validation loss
        val_nll, val_mse, val_indiv_nlls = utils.evaluate_hetvae(net,dim,val_loader,device=args.device)

        lrs.append(optimizer.param_groups[0]['lr'])

        if scheduler: scheduler.step(val_nll) 

        ######## print train / valid losses ########
        if itr % args.print_at == 0:
            print(
                '\tIter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg wnll: {:.4f}, avg kl: {:.4f}, '
                'mse: {:.6f}, wmse: {:.6f}, mae: {:.6f}, val nll: {:.4f}, val mse {:.4f}'.format(
                    itr,
                    train_loss / train_n,
                    -avg_loglik / train_n,
                    -avg_wloglik / train_n,
                    avg_kl / train_n,
                    mse / train_n,
                    wmse / train_n,
                    mae / train_n,
                    val_nll,
                    val_mse))

        ####### test loss every 10 itrs, save losses too #############
        if itr % 10 == 0:
            test_nll, test_mse, indiv_nlls = utils.evaluate_hetvae(net,dim,test_loader,device=args.device)
            train_losses.append([(-avg_loglik / train_n).item(),(mse / train_n).item(), \
                                 (avg_kl / train_n).item(),kl_coef])
            test_losses.append([test_nll.item(),test_mse.item()])
            val_losses.append([val_nll.item(),val_mse.item()])
            print('test nll: {:.4f}, test mse: {:.4f}'.format(test_nll,test_mse))

        ########### save model checkpoint ##############
        if itr % args.save_at == 0:
            print('saving.................')
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'losses': (train_losses,val_losses,test_losses),
                'lrs' : lrs
            }, lcs.folder + str((-avg_loglik / train_n).item()) + '.h5')
            print('done')

        ############# patience ####################
        if best_loss > (train_loss / train_n): 
            patience_counter += 1
        else:
            best_loss = (train_loss / train_n) 
            patience_counter == 0

        if patience_counter >= args.patience:
            print(f'training has not improved for {args.patience} epochs')
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'losses': (train_losses,val_losses,test_losses),
                'lrs': lrs
            }, lcs.folder + str((-avg_loglik / train_n).item()) + '.h5')
            print('done')
            break
            
    
@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    leaf_cfg = utils.get_leaf_nodes(cfg)
    args = argparse.Namespace(**leaf_cfg)
    args.data_folder = os.path.join(hydra.utils.get_original_cwd(), args.data_folder)
    train(args)
    
if __name__ == '__main__':
    main()
    